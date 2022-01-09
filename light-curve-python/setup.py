import toml
from setuptools import setup


def get_light_curve_version():
    with open("Cargo.toml") as fh:
        cargo_toml = toml.load(fh)
    package = cargo_toml["package"]
    version = package["version"]
    return version


def main():
    version = get_light_curve_version()
    requirement = f"light-curve=={version}"

    setup(
        version=version,
        install_requires=[requirement],
    )


if __name__ == "__main__":
    main()
