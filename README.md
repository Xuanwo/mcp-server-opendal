# Model Context Protocol Server for Apache OpenDAL™
A Model Context Protocol (MCP) server implementation that provides access to various storage services via [Apache OpenDAL™](https://opendal.apache.org/).

## Features

- Seamless access to multiple storage services including S3, Azure Blob Storage, Google Cloud Storage, and more
- List files and directories from storage services
- Read file contents with automatic text/binary detection
- Environment variable based configuration

## Installation

```shell
pip install mcp-server-opendal
```

## Usage

Configure storage services by setting environment variables. Each service requires a prefix and specific configuration options.

For example, to configure an S3 service with alias "mys3":

```
OPENDAL_MYS3_TYPE=s3
OPENDAL_MYS3_BUCKET=mybucket
OPENDAL_MYS3_REGION=us-east-1
OPENDAL_MYS3_ENDPOINT=http://localhost:9000
OPENDAL_MYS3_ACCESS_KEY_ID=myaccesskey
OPENDAL_MYS3_SECRET_ACCESS_KEY=mysecretkey
```

Then you can use tool like `read` and `list` with `mys3://path/to/file`.

`mcp-server-opendal` will also load from `.env`.

## Development

```shell
npx @modelcontextprotocol/inspector \
  uv \
  --directory . \
  run \
  mcp-server-opendal
```
