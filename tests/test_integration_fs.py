import base64
import json
import shutil
import tempfile
from pathlib import Path

import pytest

from mcp_server_opendal.server import (
    OPENDAL_OPTIONS,
    get_info,
    list,
    mcp,
    read,
    register_resources,
)


@pytest.fixture
def test_files():
    """Create a temporary directory with test files"""
    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()

    try:
        # Create some test files and directories
        root_path = Path(temp_dir)

        # Create a text file
        text_file = root_path / "test_text.txt"
        text_file.write_text(
            "This is a test text file\nSecond line content", encoding="utf-8"
        )

        # Create a binary file
        bin_file = root_path / "binary_file.bin"
        bin_file.write_bytes(b"\x00\x01\x02\x03\xff\xfe\xab\xcd")

        # Create a JSON file
        json_file = root_path / "config.json"
        json_data = {"name": "test", "value": 123, "enabled": True}
        json_file.write_text(
            json.dumps(json_data, ensure_ascii=False), encoding="utf-8"
        )

        # Create a subdirectory and a nested file
        subdir = root_path / "subdir"
        subdir.mkdir()
        nested_file = subdir / "nested_file.log"
        nested_file.write_text("Subdirectory file content", encoding="utf-8")

        yield temp_dir
    finally:
        # Clean up the temporary directory
        shutil.rmtree(temp_dir)


@pytest.fixture
def setup_env(test_files, monkeypatch):
    """Set environment variables and OpenDAL configuration"""
    OPENDAL_OPTIONS.clear()
    OPENDAL_OPTIONS.update(
        {
            "fs_type": "fs",
            "fs_root": test_files,
        }
    )

    monkeypatch.setenv("OPENDAL_FS_TYPE", "fs")
    monkeypatch.setenv("OPENDAL_FS_ROOT", test_files)

    # Re-register resources
    register_resources()

    yield


@pytest.mark.asyncio
async def test_list_resources(setup_env):
    """Test listing available resources"""
    resources = await mcp.list_resources()

    assert len(resources) > 0

    resource_schemes = [str(r.uri.scheme) for r in resources if hasattr(r, "uri")]
    assert "fs" in resource_schemes


@pytest.mark.asyncio
async def test_list_directory_contents(setup_env, test_files):
    """Test listing directory contents"""
    result = await list("fs://")

    assert "test_text.txt" in result
    assert "binary_file.bin" in result
    assert "config.json" in result
    assert "subdir" in result

    subdir_result = await list("fs://subdir/")
    assert "nested_file.log" in subdir_result


@pytest.mark.asyncio
async def test_read_text_file(setup_env):
    """Test reading a text file"""
    result = await read("fs://test_text.txt")

    assert "This is a test text file\nSecond line content" in result["content"]
    assert result["mime_type"] in ["text/plain", None]


@pytest.mark.asyncio
async def test_read_binary_file(setup_env):
    """Test reading a binary file"""
    result = await read("fs://binary_file.bin")

    expected_binary = b"\x00\x01\x02\x03\xff\xfe\xab\xcd"
    decoded = base64.b64decode(result["content"])
    assert decoded == expected_binary
    assert result.get("is_binary", False) is True


@pytest.mark.asyncio
async def test_read_json_file(setup_env):
    """Test reading a JSON file"""
    result = await read("fs://config.json")

    assert "test" in result["content"]
    assert result["mime_type"] in ["application/json", "text/plain", None]


@pytest.mark.asyncio
async def test_read_json_file_with_read(setup_env):
    """Test reading a JSON file with read"""
    result = await read("fs://config.json")

    assert "test" in result["content"]
    assert result["mime_type"] in ["application/json", "text/plain", None]


@pytest.mark.asyncio
async def test_get_file_info(setup_env):
    """Test getting file information"""
    result = await get_info("fs://test_text.txt")

    assert "test_text.txt" in result
    content = "This is a test text file\nSecond line content"
    expected_size = len(content.encode("utf-8"))
    assert f"Size: {expected_size} bytes" in result


@pytest.mark.asyncio
async def test_read_nested_file(setup_env):
    """Test reading a nested file"""
    result = await read("fs://subdir/nested_file.log")

    assert "Subdirectory file content" in result["content"]
