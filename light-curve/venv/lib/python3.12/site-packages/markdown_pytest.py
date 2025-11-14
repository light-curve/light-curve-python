from functools import partial
from itertools import groupby
from pathlib import Path
from types import CodeType
from typing import Dict, Iterable, Iterator, NamedTuple, Optional, TextIO, Tuple

import pytest


class CodeBlock(NamedTuple):
    start_line: int
    lines: Tuple[str, ...]
    arguments: Tuple[Tuple[str, str], ...]
    path: str
    name: str

    @property
    def end_line(self) -> int:
        return self.start_line + len(self.lines)


COMMENT_BRACKETS = ("<!--", "-->")


LineType = Tuple[int, str]


class LinesIterator:
    lines: Tuple[LineType, ...]

    def __init__(self, lines: Iterable[str]):
        self.lines = tuple(
            map(tuple, enumerate(line.rstrip() for line in lines)),
        )
        self.index = 0

    @classmethod
    def from_fp(cls, fp: TextIO) -> "LinesIterator":
        return cls(fp.readlines())

    @classmethod
    def from_file(cls, filename) -> "LinesIterator":
        with open(filename, "r") as fp:
            return cls.from_fp(fp)

    def get_relative(self, index: int) -> LineType:
        return self.lines[self.index + index]

    def is_last_line(self) -> bool:
        return self.index >= len(self.lines)

    def next(self) -> LineType:
        lineno, line = self.get_relative(0)
        self.index += 1
        return lineno, line

    def seek_relative(self, index: int) -> None:
        self.index += index

    def reverse_iterator(self, start_from: int = 0):
        for i in range(start_from, self.index):
            yield self.get_relative(-i - 1)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return self.next()
        except IndexError:
            raise StopIteration


def parse_arguments(line_iterator: LinesIterator) -> Dict[str, str]:

    outside_comment = inside_comment = False
    index = line_iterator.index
    # Checking if the code block is outside of the comment block
    for lineno, line in line_iterator.reverse_iterator(1):
        if not line.strip():
            continue
        if line.strip().endswith(COMMENT_BRACKETS[1]):
            outside_comment = True
        break

    # Checking if the code block is inside of the comment block
    if not outside_comment:
        for lineno, line in line_iterator:
            if not line.strip():
                continue
            if line.strip().endswith(COMMENT_BRACKETS[0]):
                return {}
            elif line.strip().endswith(COMMENT_BRACKETS[1]):
                inside_comment = True
                line_iterator.seek_relative(1)
                break

    if not outside_comment and not inside_comment:
        return {}

    lines = []
    reverse_iterator = line_iterator.reverse_iterator(1)
    for lineno, line in reverse_iterator:
        if line.strip().startswith("```"):
            for _, line in reverse_iterator:
                if line.strip().startswith("```"):
                    break
            continue
        lines.append(line)
        if line.strip().startswith(COMMENT_BRACKETS[0]):
            break

    # Restore the iterator (due to inside comment forward iterations)
    line_iterator.index = index

    if not lines:
        return {}

    lines = lines[::-1]
    result = {}
    args = "".join(
        "".join(lines).strip()[
            len(COMMENT_BRACKETS[0]):-len(COMMENT_BRACKETS[1]) + 1
        ].strip("-").strip().splitlines(),
    ).split(";")

    for arg in args:
        if ":" not in arg:
            continue

        key, value = arg.split(":", 1)
        result[key.strip()] = value.strip()

    return result


def parse_code_blocks(fspath) -> Iterator[CodeBlock]:
    line_iterator = LinesIterator.from_file(fspath)

    for lineno, line in line_iterator:
        if (
            line.rstrip().endswith("```") and
            line.lstrip().startswith("```")
        ):
            # skip all blocks without '```python`
            end_of_block = "`" * line.count("`")
            try:
                lineno, line = line_iterator.next()
            except IndexError:
                return

            for lineno, line in line_iterator:
                if line.rstrip() == end_of_block:
                    break

        if not line.endswith("```python"):
            continue

        indent = line.rstrip().count(" ")
        end_of_block = (" " * indent) + ("`" * line.count("`"))

        arguments = parse_arguments(line_iterator)

        # the next line after ```python
        start_lineno = lineno + 1
        code_lines = []

        for lineno, line in line_iterator:
            if line.startswith(end_of_block):
                break
            code_lines.append(line[indent:])

        if not arguments or "name" not in arguments:
            continue

        case = arguments.get("case")
        if case is not None:
            start_lineno -= 1
            # indent test case lines
            code_lines = [f"    {code_line}" for code_line in code_lines]
            code_lines.insert(
                0, "with __markdown_pytest_subtests_fixture.test("
                   f"msg='{case} line={start_lineno}'):",
            )

        block = CodeBlock(
            start_line=start_lineno,
            lines=tuple(code_lines),
            arguments=tuple(arguments.items()),
            path=str(fspath),
            name=arguments.pop("name"),
        )

        yield block


def compile_code_blocks(*blocks: CodeBlock) -> Optional[CodeType]:
    blocks = sorted(blocks, key=lambda x: x.start_line)
    if not blocks:
        return None
    lines = [""] * blocks[-1].end_line
    path = blocks[0].path
    for block in blocks:
        lines[block.start_line:block.end_line] = block.lines
    return compile(source="\n".join(lines), mode="exec", filename=path)


class MDModule(pytest.Module):

    @staticmethod
    def caller(code, subtests):
        eval(code, dict(__markdown_pytest_subtests_fixture=subtests))

    def collect(self) -> Iterable[pytest.Function]:
        test_prefix = self.config.getoption("--md-prefix")

        for test_name, blocks in groupby(
            parse_code_blocks(self.fspath),
            key=lambda x: x.name,
        ):
            if not test_name.startswith(test_prefix):
                continue

            blocks = list(blocks)
            code = compile_code_blocks(*blocks)
            if code is None:
                continue

            yield pytest.Function.from_parent(
                name=test_name,
                parent=self,
                callobj=partial(self.caller, code),
            )


def pytest_addoption(parser):
    parser.addoption(
        "--md-prefix", default="test",
        help="Markdown test code-block prefix from comment",
    )


@pytest.hookimpl(trylast=True)
def pytest_collect_file(path, parent: pytest.Collector) -> Optional[MDModule]:
    if path.ext.lower() not in (".md", ".markdown"):
        return None
    return MDModule.from_parent(parent=parent, path=Path(path))
