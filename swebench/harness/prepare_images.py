import docker
import resource

from argparse import ArgumentParser
from pathlib import Path

from swebench.harness.build_state import BuildState
from swebench.harness.constants import KEY_INSTANCE_ID
from swebench.harness.docker_build import build_env_images, build_instance_images
from swebench.harness.docker_utils import list_images
from swebench.harness.test_spec.test_spec import make_test_spec
from swebench.harness.utils import load_swebench_dataset, str2bool, optional_str


def filter_dataset_to_build(
    dataset: list,
    instance_ids: list | None,
    client: docker.DockerClient,
    force_rebuild: bool,
    namespace: str = None,
    tag: str = None,
    env_image_tag: str = None,
    arch: str = "x86_64",
):
    """
    Filter the dataset to only include instances that need to be built.

    Args:
        dataset (list): List of instances (usually all of SWE-bench dev/test split)
        instance_ids (list): List of instance IDs to build.
        client (docker.DockerClient): Docker client.
        force_rebuild (bool): Whether to force rebuild all images.
    """
    existing_images = list_images(client)
    data_to_build = []

    if instance_ids is None:
        instance_ids = [instance[KEY_INSTANCE_ID] for instance in dataset]

    not_in_dataset = set(instance_ids).difference(
        set([instance[KEY_INSTANCE_ID] for instance in dataset])
    )
    if not_in_dataset:
        raise ValueError(f"Instance IDs not found in dataset: {not_in_dataset}")

    for instance in dataset:
        if instance[KEY_INSTANCE_ID] not in instance_ids:
            continue

        kwargs = dict(namespace=namespace, arch=arch)
        if tag is not None:
            kwargs["instance_image_tag"] = tag
        if env_image_tag is not None:
            kwargs["env_image_tag"] = env_image_tag
        spec = make_test_spec(instance, **kwargs)
        if force_rebuild:
            data_to_build.append(instance)
        elif spec.instance_image_key not in existing_images:
            data_to_build.append(instance)

    return data_to_build


def main(
    dataset_name,
    split,
    instance_ids,
    max_workers,
    force_rebuild,
    open_file_limit,
    namespace,
    tag,
    env_image_tag,
    arch,
    state_file,
    retry_failed,
    registry,
    env_only=False,
    verify=False,
    verify_timeout=300,
):
    """
    Build Docker images for the specified instances.

    Args:
        instance_ids (list): List of instance IDs to build.
        max_workers (int): Number of workers for parallel processing.
        force_rebuild (bool): Whether to force rebuild all images.
        open_file_limit (int): Open file limit.
        arch (str): Target architecture (x86_64 or arm64).
        state_file (str): Path to JSON state file for tracking build progress.
        retry_failed (bool): If True, only retry previously failed instances.
        registry (str): Registry path to push images to after building.
        verify (bool): Run gold patch eval before pushing to catch broken images.
        verify_timeout (int): Timeout in seconds for the verification eval script.
    """
    resource.setrlimit(resource.RLIMIT_NOFILE, (open_file_limit, open_file_limit))
    client = docker.from_env()

    dataset = load_swebench_dataset(dataset_name, split)

    build_state = None
    if state_file:
        build_state = BuildState(Path(state_file), dataset_name, arch)
        all_ids = [inst[KEY_INSTANCE_ID] for inst in dataset]
        build_state.initialize(all_ids)

        if retry_failed:
            instance_ids = build_state.get_failed()
            if not instance_ids:
                print("No failed instances to retry.")
                return 0
            build_state.reset_status(instance_ids)
            print(f"Retrying {len(instance_ids)} previously failed instances")
        elif not force_rebuild:
            successful = set(build_state.get_successful())
            if instance_ids is not None:
                instance_ids = [i for i in instance_ids if i not in successful]
            else:
                instance_ids = [i for i in all_ids if i not in successful]
            if not instance_ids:
                print("All images already built (per state file). Nothing to do.")
                return 0

    dataset = filter_dataset_to_build(
        dataset, instance_ids, client, force_rebuild, namespace, tag, env_image_tag,
        arch=arch,
    )

    if len(dataset) == 0:
        print("All images exist. Nothing left to build.")
        return 0

    if env_only:
        from swebench.harness.test_spec.test_spec import get_test_specs_from_dataset
        print(f"Building env images only for {len(dataset)} instances (arch={arch})")
        spec_kwargs = dict(namespace=namespace, arch=arch)
        if tag is not None:
            spec_kwargs["instance_image_tag"] = tag
        if env_image_tag is not None:
            spec_kwargs["env_image_tag"] = env_image_tag
        test_specs = list(
            map(lambda x: make_test_spec(x, **spec_kwargs), dataset)
        )
        env_successful, env_failed = build_env_images(
            client, test_specs, force_rebuild, max_workers, arch=arch,
        )
        print(f"Env images: {len(env_successful)} succeeded, {len(env_failed)} failed")
        if env_failed:
            print("Failed env images:")
            for payload in env_failed:
                print(f"  {payload[0]}")
        return

    print(f"Building {len(dataset)} instance images (arch={arch})")

    successful, failed = build_instance_images(
        client=client,
        dataset=dataset,
        force_rebuild=force_rebuild,
        max_workers=max_workers,
        namespace=namespace,
        tag=tag,
        env_image_tag=env_image_tag,
        arch=arch,
        build_state=build_state,
        registry=registry,
        verify=verify,
        verify_timeout=verify_timeout,
    )
    print(f"Successfully built {len(successful)} images")
    print(f"Failed to build {len(failed)} images")

    if build_state:
        summary = build_state.summary()
        print(f"\nBuild state summary: {summary}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="SWE-bench/SWE-bench_Lite",
        help="Name of the dataset to use",
    )
    parser.add_argument("--split", type=str, default="test", help="Split to use")
    parser.add_argument(
        "--instance_ids",
        nargs="+",
        type=str,
        help="Instance IDs to run (space separated)",
    )
    parser.add_argument(
        "--instance_ids_file",
        type=str,
        default=None,
        help="Path to a text file with one instance ID per line",
    )
    parser.add_argument(
        "--max_workers", type=int, default=4, help="Max workers for parallel processing"
    )
    parser.add_argument(
        "--force_rebuild", type=str2bool, default=False, help="Force rebuild images"
    )
    parser.add_argument(
        "--open_file_limit", type=int, default=8192, help="Open file limit"
    )
    parser.add_argument(
        "--namespace",
        type=optional_str,
        default=None,
        help="Namespace to use for the images (default: None)",
    )
    parser.add_argument(
        "--tag", type=str, default=None, help="Tag to use for the images"
    )
    parser.add_argument(
        "--env_image_tag", type=str, default=None, help="Environment image tag to use"
    )
    parser.add_argument(
        "--arch",
        type=str,
        default="x86_64",
        choices=["x86_64", "arm64"],
        help="Target architecture for images",
    )
    parser.add_argument(
        "--state_file",
        type=str,
        default=None,
        help="Path to JSON state file for tracking build progress",
    )
    parser.add_argument(
        "--retry_failed",
        type=str2bool,
        default=False,
        help="Retry only previously failed instances (requires --state_file)",
    )
    parser.add_argument(
        "--registry",
        type=str,
        default=None,
        help="Registry path to push images to (e.g. registry.example.com/namespace/swe-bench)",
    )
    parser.add_argument(
        "--env_only",
        type=str2bool,
        default=False,
        help="Only build base + env images (skip instance images)",
    )
    parser.add_argument(
        "--verify",
        type=str2bool,
        default=False,
        help="Run gold patch evaluation after building each image to verify correctness before pushing",
    )
    parser.add_argument(
        "--verify_timeout",
        type=int,
        default=300,
        help="Timeout in seconds for verification eval script (default: 300)",
    )
    args = parser.parse_args()
    if args.instance_ids_file:
        with open(args.instance_ids_file) as f:
            file_ids = [l.strip() for l in f if l.strip()]
        if args.instance_ids:
            args.instance_ids.extend(file_ids)
        else:
            args.instance_ids = file_ids
    del args.instance_ids_file
    main(**vars(args))
