from __future__ import annotations

import docker
import docker.errors
import logging
import subprocess
import threading
import sys
import time
import traceback

from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory

from swebench.harness.build_state import BuildState
from swebench.harness.constants import (
    BASE_IMAGE_BUILD_DIR,
    DOCKER_PATCH,
    DOCKER_USER,
    DOCKER_WORKDIR,
    ENV_IMAGE_BUILD_DIR,
    INSTANCE_IMAGE_BUILD_DIR,
    KEY_INSTANCE_ID,
    KEY_MODEL,
    KEY_PREDICTION,
    UTF8,
)
from swebench.harness.docker_utils import (
    cleanup_container,
    remove_image,
)
from swebench.harness.grading import get_eval_report
from swebench.harness.test_spec.test_spec import (
    get_test_specs_from_dataset,
    make_test_spec,
    TestSpec,
)
from swebench.harness.utils import ansi_escape, run_threadpool


class BuildImageError(Exception):
    def __init__(self, image_name, message, logger):
        super().__init__(message)
        self.super_str = super().__str__()
        self.image_name = image_name
        self.log_path = logger.log_file
        self.logger = logger

    def __str__(self):
        return (
            f"Error building image {self.image_name}: {self.super_str}\n"
            f"Check ({self.log_path}) for more information."
        )


def setup_logger(instance_id: str, log_file: Path, mode="w", add_stdout: bool = False):
    """
    This logger is used for logging the build process of images and containers.
    It writes logs to the log file.

    If `add_stdout` is True, logs will also be sent to stdout, which can be used for
    streaming ephemeral output from Modal containers.
    """
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"{instance_id}.{log_file.name}")
    handler = logging.FileHandler(log_file, mode=mode, encoding=UTF8)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    setattr(logger, "log_file", log_file)
    if add_stdout:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            f"%(asctime)s - {instance_id} - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def close_logger(logger):
    # To avoid too many open files
    for handler in logger.handlers:
        handler.close()
        logger.removeHandler(handler)


def build_image(
    image_name: str,
    setup_scripts: dict,
    dockerfile: str,
    platform: str,
    client: docker.DockerClient,
    build_dir: Path,
    nocache: bool = False,
):
    """
    Builds a docker image with the given name, setup scripts, dockerfile, and platform.

    Args:
        image_name (str): Name of the image to build
        setup_scripts (dict): Dictionary of setup script names to setup script contents
        dockerfile (str): Contents of the Dockerfile
        platform (str): Platform to build the image for
        client (docker.DockerClient): Docker client to use for building the image
        build_dir (Path): Directory for the build context (will also contain logs, scripts, and artifacts)
        nocache (bool): Whether to use the cache when building
    """
    # Create a logger for the build process
    logger = setup_logger(image_name, build_dir / "build_image.log")
    logger.info(
        f"Building image {image_name}\n"
        f"Using dockerfile:\n{dockerfile}\n"
        f"Adding ({len(setup_scripts)}) setup scripts to image build repo"
    )

    for setup_script_name, setup_script in setup_scripts.items():
        logger.info(f"[SETUP SCRIPT] {setup_script_name}:\n{setup_script}")
    try:
        # Write the setup scripts to the build directory
        for setup_script_name, setup_script in setup_scripts.items():
            setup_script_path = build_dir / setup_script_name
            with open(setup_script_path, "w") as f:
                f.write(setup_script)
            if setup_script_name not in dockerfile:
                logger.warning(
                    f"Setup script {setup_script_name} may not be used in Dockerfile"
                )

        # Write the dockerfile to the build directory
        dockerfile_path = build_dir / "Dockerfile"
        with open(dockerfile_path, "w") as f:
            f.write(dockerfile)

        # Build the image
        logger.info(
            f"Building docker image {image_name} in {build_dir} with platform {platform}"
        )
        response = client.api.build(
            path=str(build_dir),
            tag=image_name,
            rm=True,
            forcerm=True,
            decode=True,
            platform=platform,
            nocache=nocache,
        )

        # Log the build process continuously
        buildlog = ""
        for chunk in response:
            if "stream" in chunk:
                # Remove ANSI escape sequences from the log
                chunk_stream = ansi_escape(chunk["stream"])
                logger.info(chunk_stream.strip())
                buildlog += chunk_stream
            elif "errorDetail" in chunk:
                # Decode error message, raise BuildError
                logger.error(f"Error: {ansi_escape(chunk['errorDetail']['message'])}")
                raise docker.errors.BuildError(
                    chunk["errorDetail"]["message"], buildlog
                )
        logger.info("Image built successfully!")
    except docker.errors.BuildError as e:
        logger.error(f"docker.errors.BuildError during {image_name}: {e}")
        raise BuildImageError(image_name, str(e), logger) from e
    except Exception as e:
        logger.error(f"Error building image {image_name}: {e}")
        raise BuildImageError(image_name, str(e), logger) from e
    finally:
        close_logger(logger)  # functions that create loggers should close them


def build_base_images(
    client: docker.DockerClient,
    dataset: list,
    force_rebuild: bool = False,
    namespace: str = None,
    instance_image_tag: str = None,
    env_image_tag: str = None,
    arch: str = "x86_64",
):
    """
    Builds the base images required for the dataset if they do not already exist.

    Args:
        client (docker.DockerClient): Docker client to use for building the images
        dataset (list): List of test specs or dataset to build images for
        force_rebuild (bool): Whether to force rebuild the images even if they already exist
    """
    # Get the base images to build from the dataset
    test_specs = get_test_specs_from_dataset(
        dataset,
        namespace=namespace,
        instance_image_tag=instance_image_tag,
        env_image_tag=env_image_tag,
        arch=arch,
    )
    base_images = {
        x.base_image_key: (x.base_dockerfile, x.platform) for x in test_specs
    }

    # Build the base images
    for image_name, (dockerfile, platform) in base_images.items():
        try:
            # Check if the base image already exists
            client.images.get(image_name)
            if force_rebuild:
                # Remove the base image if it exists and force rebuild is enabled
                remove_image(client, image_name, "quiet")
            else:
                print(f"Base image {image_name} already exists, skipping build.")
                continue
        except docker.errors.ImageNotFound:
            pass
        # Build the base image (if it does not exist or force rebuild is enabled)
        print(f"Building base image ({image_name})")
        build_image(
            image_name=image_name,
            setup_scripts={},
            dockerfile=dockerfile,
            platform=platform,
            client=client,
            build_dir=BASE_IMAGE_BUILD_DIR / image_name.replace(":", "__"),
        )
    print("Base images built successfully.")


def get_env_configs_to_build(
    client: docker.DockerClient,
    dataset: list,
    namespace: str = None,
    instance_image_tag: str = None,
    env_image_tag: str = None,
    arch: str = "x86_64",
):
    """
    Returns a dictionary of image names to build scripts and dockerfiles for environment images.
    Returns only the environment images that need to be built.

    Args:
        client (docker.DockerClient): Docker client to use for building the images
        dataset (list): List of test specs or dataset to build images for
    """
    image_scripts = dict()
    base_images = dict()
    test_specs = get_test_specs_from_dataset(
        dataset,
        namespace=namespace,
        instance_image_tag=instance_image_tag,
        env_image_tag=env_image_tag,
        arch=arch,
    )

    for test_spec in test_specs:
        # Check if the base image exists
        try:
            if test_spec.base_image_key not in base_images:
                base_images[test_spec.base_image_key] = client.images.get(
                    test_spec.base_image_key
                )
            base_image = base_images[test_spec.base_image_key]
        except docker.errors.ImageNotFound:
            raise Exception(
                f"Base image {test_spec.base_image_key} not found for {test_spec.env_image_key}\n."
                "Please build the base images first."
            )

        # Check if the environment image exists
        image_exists = False
        try:
            env_image = client.images.get(test_spec.env_image_key)
            image_exists = True
        except docker.errors.ImageNotFound:
            pass
        if not image_exists:
            # Add the environment image to the list of images to build
            image_scripts[test_spec.env_image_key] = {
                "setup_script": test_spec.setup_env_script,
                "dockerfile": test_spec.env_dockerfile,
                "platform": test_spec.platform,
            }
    return image_scripts


def build_env_images(
    client: docker.DockerClient,
    dataset: list,
    force_rebuild: bool = False,
    max_workers: int = 4,
    namespace: str = None,
    instance_image_tag: str = None,
    env_image_tag: str = None,
    arch: str = "x86_64",
):
    """
    Builds the environment images required for the dataset if they do not already exist.

    Args:
        client (docker.DockerClient): Docker client to use for building the images
        dataset (list): List of test specs or dataset to build images for
        force_rebuild (bool): Whether to force rebuild the images even if they already exist
        max_workers (int): Maximum number of workers to use for building images
    """
    # Get the environment images to build from the dataset
    if force_rebuild:
        env_image_keys = {
            x.env_image_key
            for x in get_test_specs_from_dataset(
                dataset,
                namespace=namespace,
                instance_image_tag=instance_image_tag,
                env_image_tag=env_image_tag,
                arch=arch,
            )
        }
        for key in env_image_keys:
            remove_image(client, key, "quiet")
    build_base_images(
        client, dataset, force_rebuild, namespace, instance_image_tag, env_image_tag,
        arch=arch,
    )
    configs_to_build = get_env_configs_to_build(
        client, dataset, namespace, instance_image_tag, env_image_tag, arch=arch,
    )
    if len(configs_to_build) == 0:
        print("No environment images need to be built.")
        return [], []
    print(f"Total environment images to build: {len(configs_to_build)}")

    args_list = list()
    for image_name, config in configs_to_build.items():
        args_list.append(
            (
                image_name,
                {"setup_env.sh": config["setup_script"]},
                config["dockerfile"],
                config["platform"],
                client,
                ENV_IMAGE_BUILD_DIR / image_name.replace(":", "__"),
            )
        )

    env_total = len(args_list)
    _env_lock = threading.Lock()
    _env_progress = {"done": 0}

    def _build_env_with_progress(image_name, *rest):
        try:
            build_image(image_name, *rest)
        except Exception:
            with _env_lock:
                _env_progress["done"] += 1
                print(f"  [env {_env_progress['done']}/{env_total}] FAIL {image_name}", flush=True)
            raise
        else:
            with _env_lock:
                _env_progress["done"] += 1
                print(f"  [env {_env_progress['done']}/{env_total}] OK   {image_name}", flush=True)

    successful, failed = run_threadpool(_build_env_with_progress, args_list, max_workers)
    if len(failed) == 0:
        print("All environment images built successfully.")
    else:
        print(f"{len(failed)} environment images failed to build.")

    # Return the list of (un)successfuly built images
    return successful, failed


def build_instance_images(
    client: docker.DockerClient,
    dataset: list,
    force_rebuild: bool = False,
    max_workers: int = 4,
    namespace: str = None,
    tag: str = None,
    env_image_tag: str = None,
    arch: str = "x86_64",
    build_state: BuildState | None = None,
    registry: str | None = None,
    verify: bool = False,
    verify_timeout: int = 300,
):
    """
    Builds the instance images required for the dataset if they do not already exist.

    Args:
        dataset (list): List of test specs or dataset to build images for
        client (docker.DockerClient): Docker client to use for building the images
        force_rebuild (bool): Whether to force rebuild the images even if they already exist
        max_workers (int): Maximum number of workers to use for building images
        build_state (BuildState | None): Optional state tracker for recording build results
        registry (str | None): Registry path to push images to
        verify (bool): If True, run gold patch verification before pushing
        verify_timeout (int): Timeout in seconds for verification eval
    """
    # Build environment images (and base images as needed) first
    spec_kwargs = dict(namespace=namespace, arch=arch)
    if tag is not None:
        spec_kwargs["instance_image_tag"] = tag
    if env_image_tag is not None:
        spec_kwargs["env_image_tag"] = env_image_tag
    test_specs = list(
        map(
            lambda x: make_test_spec(x, **spec_kwargs),
            dataset,
        )
    )
    if force_rebuild:
        for spec in test_specs:
            remove_image(client, spec.instance_image_key, "quiet")
    _, env_failed = build_env_images(
        client, test_specs, force_rebuild, max_workers, arch=arch,
    )

    if len(env_failed) > 0:
        # env_failed payloads are tuples where the first element is the image name
        env_failed_keys = {payload[0] for payload in env_failed}
        dont_run_specs = [
            spec for spec in test_specs if spec.env_image_key in env_failed_keys
        ]
        test_specs = [
            spec for spec in test_specs if spec.env_image_key not in env_failed_keys
        ]
        if build_state:
            for spec in dont_run_specs:
                build_state.mark_env_failed(
                    spec.instance_id, f"env image {spec.env_image_key} failed to build",
                )
        print(
            f"Skipping {len(dont_run_specs)} instances - due to failed env image builds"
        )
    gold_patches = {}
    if verify:
        gold_patches = {
            inst[KEY_INSTANCE_ID]: inst.get("patch", "") for inst in dataset
        }

    total = len(test_specs)
    verb = "Building + verifying" if verify else "Building"
    print(f"{verb} instance images for {total} instances")
    successful, failed = list(), list()

    _progress_lock = threading.Lock()
    _progress = {"done": 0, "ok": 0, "fail": 0, "total": total}

    def _build_with_progress(
        spec, client, logger, nocache, build_state, registry,
        verify, gold_patch, verify_timeout,
    ):
        try:
            build_instance_image(
                spec, client, logger, nocache, build_state, registry,
                verify=verify, gold_patch=gold_patch,
                verify_timeout=verify_timeout,
            )
        except Exception as exc:
            label = "VFAIL" if "Verification failed" in str(exc) else "FAIL"
            with _progress_lock:
                _progress["done"] += 1
                _progress["fail"] += 1
                print(
                    f"  [{_progress['done']}/{_progress['total']}] {label} {spec.instance_id}",
                    flush=True,
                )
            raise
        else:
            with _progress_lock:
                _progress["done"] += 1
                _progress["ok"] += 1
                print(
                    f"  [{_progress['done']}/{_progress['total']}] OK   {spec.instance_id}",
                    flush=True,
                )

    payloads = [
        (
            spec, client, None, False, build_state, registry,
            verify, gold_patches.get(spec.instance_id), verify_timeout,
        )
        for spec in test_specs
    ]
    successful, failed = run_threadpool(_build_with_progress, payloads, max_workers)
    # Show how many images failed to build
    if len(failed) == 0:
        print("All instance images built successfully.")
    else:
        print(f"{len(failed)} instance images failed to build.")

    # Return the list of (un)successfuly built images
    return successful, failed


def build_instance_image(
    test_spec: TestSpec,
    client: docker.DockerClient,
    logger: logging.Logger | None,
    nocache: bool,
    build_state: BuildState | None = None,
    registry: str | None = None,
    verify: bool = False,
    gold_patch: str | None = None,
    verify_timeout: int = 300,
):
    """
    Builds the instance image for the given test spec if it does not already exist.
    Optionally verifies the image by running the gold patch, then pushes to a registry.

    Args:
        test_spec (TestSpec): Test spec to build the instance image for
        client (docker.DockerClient): Docker client to use for building the image
        logger (logging.Logger): Logger to use for logging the build process
        nocache (bool): Whether to use the cache when building
        build_state (BuildState | None): Optional state tracker for recording build results
        registry (str | None): Registry path to push to
        verify (bool): If True, run gold patch evaluation before pushing
        gold_patch (str | None): The gold patch content (required when verify=True)
        verify_timeout (int): Timeout in seconds for the verification eval script
    """
    instance_id = test_spec.instance_id
    if build_state:
        build_state.mark_building(
            instance_id, test_spec.instance_image_key, test_spec.env_image_key,
        )
    start = time.time()

    build_ok = False

    # Set up logging for the build process
    build_dir = INSTANCE_IMAGE_BUILD_DIR / test_spec.instance_image_key.replace(
        ":", "__"
    )
    new_logger = False
    if logger is None:
        new_logger = True
        logger = setup_logger(test_spec.instance_id, build_dir / "prepare_image.log")

    # Get the image names and dockerfile for the instance image
    image_name = test_spec.instance_image_key
    env_image_name = test_spec.env_image_key
    dockerfile = test_spec.instance_dockerfile

    try:
        # Check that the env. image the instance image is based on exists
        try:
            env_image = client.images.get(env_image_name)
        except docker.errors.ImageNotFound as e:
            raise BuildImageError(
                test_spec.instance_id,
                f"Environment image {env_image_name} not found for {test_spec.instance_id}",
                logger,
            ) from e
        logger.info(
            f"Environment image {env_image_name} found for {test_spec.instance_id}\n"
            f"Building instance image {image_name} for {test_spec.instance_id}"
        )

        # Check if the instance image already exists
        image_exists = False
        try:
            client.images.get(image_name)
            image_exists = True
        except docker.errors.ImageNotFound:
            pass

        # Build the instance image
        if not image_exists:
            build_image(
                image_name=image_name,
                setup_scripts={
                    "setup_repo.sh": test_spec.install_repo_script,
                },
                dockerfile=dockerfile,
                platform=test_spec.platform,
                client=client,
                build_dir=build_dir,
                nocache=nocache,
            )
        else:
            logger.info(f"Image {image_name} already exists, skipping build.")

        if build_state:
            build_state.mark_success(instance_id, time.time() - start)
        build_ok = True

        if verify and gold_patch is not None:
            resolved, verify_err = _verify_gold_patch(
                test_spec, client, gold_patch, logger, verify_timeout,
            )
            if not resolved:
                if build_state:
                    build_state.mark_verify_failed(
                        instance_id, verify_err or "gold patch did not resolve",
                    )
                raise BuildImageError(
                    instance_id,
                    f"Verification failed: {verify_err or 'gold patch did not resolve'}",
                    logger,
                )
            if build_state:
                build_state.mark_verified(instance_id)

        if registry:
            _push_to_registry(
                client, image_name, registry, instance_id, logger, build_state,
            )
    except Exception as e:
        if build_state and not build_ok:
            build_state.mark_failed(instance_id, str(e), time.time() - start)
        raise
    finally:
        if new_logger:
            close_logger(logger)


def _subprocess_docker(cmd: list[str], logger: logging.Logger, timeout: int | None = None) -> subprocess.CompletedProcess:
    """Run a docker CLI command via subprocess."""
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def _verify_gold_patch(
    test_spec: TestSpec,
    client: docker.DockerClient,
    gold_patch: str,
    logger: logging.Logger,
    timeout: int = 300,
) -> tuple[bool, str | None]:
    """Run the gold patch against the instance image and check if tests resolve.

    Uses subprocess for all Docker operations to avoid the Docker Python SDK's
    shared HTTP connection pool, which drops connections under concurrent load.

    Returns (resolved, error_message).  error_message is None on success.
    """
    instance_id = test_spec.instance_id
    run_id = f"verify_{int(time.time())}"
    container_name = test_spec.get_instance_container_name(run_id)

    try:
        platform_args = ["--platform", test_spec.platform] if test_spec.platform else []

        result = _subprocess_docker(
            ["docker", "create"] + platform_args + [
                "--name", container_name,
                "--user", DOCKER_USER,
                "--cpuset-mems=0",
                test_spec.instance_image_key,
                "tail", "-f", "/dev/null",
            ],
            logger,
        )
        if result.returncode != 0:
            return False, f"docker create failed: {result.stderr.strip()}"

        result = _subprocess_docker(["docker", "start", container_name], logger)
        if result.returncode != 0:
            return False, f"docker start failed: {result.stderr.strip()}"
        logger.info(f"Verify container started for {instance_id}: {container_name}")

        with TemporaryDirectory() as tmpdir:
            patch_file = Path(tmpdir) / "patch.diff"
            patch_file.write_text(gold_patch or "")
            result = _subprocess_docker(
                ["docker", "cp", str(patch_file), f"{container_name}:{DOCKER_PATCH}"],
                logger,
            )
            if result.returncode != 0:
                return False, f"docker cp patch failed: {result.stderr.strip()}"

        applied = False
        for cmd in [
            f"git apply --verbose {DOCKER_PATCH}",
            f"patch --batch --fuzz=5 -p1 -i {DOCKER_PATCH}",
        ]:
            result = _subprocess_docker(
                ["docker", "exec", "-w", DOCKER_WORKDIR, "-u", DOCKER_USER, container_name, "bash", "-c", cmd],
                logger,
            )
            if result.returncode == 0:
                logger.info(f"Verify: patch applied with '{cmd.split()[0]}'")
                applied = True
                break
        if not applied:
            msg = f"Could not apply gold patch: {result.stderr[:500]}"
            logger.error(f"Verify: {msg}")
            return False, msg

        with TemporaryDirectory() as tmpdir:
            eval_file = Path(tmpdir) / "eval.sh"
            eval_file.write_text(test_spec.eval_script)
            result = _subprocess_docker(
                ["docker", "cp", str(eval_file), f"{container_name}:/eval.sh"],
                logger,
            )
            if result.returncode != 0:
                return False, f"docker cp eval failed: {result.stderr.strip()}"

        start_time = time.time()
        try:
            result = subprocess.run(
                ["docker", "exec", "-u", DOCKER_USER, container_name, "/bin/bash", "/eval.sh"],
                capture_output=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, timeout=timeout,
            )
            timed_out = False
        except subprocess.TimeoutExpired:
            timed_out = True
        runtime = time.time() - start_time

        logger.info(f"Verify: eval finished in {runtime:.1f}s (timed_out={timed_out})")

        if timed_out:
            return False, f"Eval timed out after {timeout}s"

        test_output = result.stdout

        build_dir = INSTANCE_IMAGE_BUILD_DIR / test_spec.instance_image_key.replace(":", "__")
        build_dir.mkdir(parents=True, exist_ok=True)
        verify_log = build_dir / "verify_eval.log"
        verify_log.write_text(test_output)
        logger.info(f"Verify: eval output saved to {verify_log}")

        pred = {
            KEY_INSTANCE_ID: instance_id,
            KEY_MODEL: "gold",
            KEY_PREDICTION: gold_patch,
        }
        report = get_eval_report(
            test_spec=test_spec,
            prediction=pred,
            test_log_path=verify_log,
            include_tests_status=True,
        )

        instance_report = report.get(instance_id, {})
        resolved = instance_report.get("resolved", False)
        tests_status = instance_report.get("tests_status", {})

        f2p = tests_status.get("FAIL_TO_PASS", {})
        p2p = tests_status.get("PASS_TO_PASS", {})
        f2p_pass = len(f2p.get("success", []))
        f2p_fail = len(f2p.get("failure", []))
        p2p_fail_list = p2p.get("failure", [])

        logger.info(
            f"Verify: resolved={resolved}  "
            f"FAIL_TO_PASS={f2p_pass}/{f2p_pass + f2p_fail}  "
            f"PASS_TO_PASS_failures={len(p2p_fail_list)}"
        )

        if f2p_fail > 0:
            return False, f"FAIL_TO_PASS not satisfied ({f2p_pass}/{f2p_pass + f2p_fail})"

        if p2p_fail_list:
            return False, f"PASS_TO_PASS regressions ({len(p2p_fail_list)}): {p2p_fail_list}"

        return True, None

    except Exception as e:
        msg = f"Verification error: {e}"
        logger.error(msg)
        return False, msg
    finally:
        subprocess.run(
            ["docker", "rm", "-f", container_name],
            capture_output=True, timeout=30,
        )


def _push_to_registry(
    client: docker.DockerClient,
    local_image_name: str,
    registry: str,
    instance_id: str,
    logger: logging.Logger,
    build_state: BuildState | None = None,
):
    """Tag a locally built image and push it to a remote container registry."""
    registry_tag = instance_id.lower()
    remote_image = f"{registry}:{registry_tag}"
    try:
        image = client.images.get(local_image_name)
        image.tag(registry, tag=registry_tag)
        logger.info(f"Pushing {remote_image}")

        for line in client.api.push(registry, tag=registry_tag, stream=True, decode=True):
            if "status" in line:
                logger.info(f"  push: {line['status']} {line.get('progress', '')}")
            if "error" in line:
                raise docker.errors.APIError(line["error"])

        logger.info(f"Successfully pushed {remote_image}")
        if build_state:
            build_state.mark_pushed(instance_id, remote_image)
    except Exception as e:
        logger.error(f"Failed to push {remote_image}: {e}")
        if build_state:
            build_state.mark_push_failed(instance_id, str(e))
        raise


def build_container(
    test_spec: TestSpec,
    client: docker.DockerClient,
    run_id: str,
    logger: logging.Logger,
    nocache: bool,
    force_rebuild: bool = False,
):
    """
    Builds the instance image for the given test spec and creates a container from the image.

    Args:
        test_spec (TestSpec): Test spec to build the instance image and container for
        client (docker.DockerClient): Docker client for building image + creating the container
        run_id (str): Run ID identifying process, used for the container name
        logger (logging.Logger): Logger to use for logging the build process
        nocache (bool): Whether to use the cache when building
        force_rebuild (bool): Whether to force rebuild the image even if it already exists
    """
    # Build corresponding instance image
    if force_rebuild:
        remove_image(client, test_spec.instance_image_key, "quiet")
    if not test_spec.is_remote_image:
        build_instance_image(test_spec, client, logger, nocache)
    else:
        try:
            client.images.get(test_spec.instance_image_key)
        except docker.errors.ImageNotFound:
            try:
                client.images.pull(test_spec.instance_image_key)
            except docker.errors.NotFound as e:
                raise BuildImageError(test_spec.instance_id, str(e), logger) from e
            except Exception as e:
                raise Exception(
                    f"Error occurred while pulling image {test_spec.base_image_key}: {str(e)}"
                )

    container = None
    try:
        # Create the container
        logger.info(f"Creating container for {test_spec.instance_id}...")

        # Define arguments for running the container
        run_args = test_spec.docker_specs.get("run_args", {})
        cap_add = run_args.get("cap_add", [])

        container = client.containers.create(
            image=test_spec.instance_image_key,
            name=test_spec.get_instance_container_name(run_id),
            user=DOCKER_USER,
            detach=True,
            command="tail -f /dev/null",
            platform=test_spec.platform,
            cap_add=cap_add,
        )
        logger.info(f"Container for {test_spec.instance_id} created: {container.id}")
        return container
    except Exception as e:
        # If an error occurs, clean up the container and raise an exception
        logger.error(f"Error creating container for {test_spec.instance_id}: {e}")
        logger.info(traceback.format_exc())
        cleanup_container(client, container, logger)
        raise BuildImageError(test_spec.instance_id, str(e), logger) from e
