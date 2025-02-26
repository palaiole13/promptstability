# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]
- Upcoming changes will be listed here.

## [0.1.4] - 2025-02-26
### Added
- Changed get_openai_api_key() to the more flexible get_api_key() which works with
openai, mistral, anthropic (claude), cohere and huggingface
- Added docstring documentation in core.py
- Added API Documentation link
- Updated README.md to include interoperable examples (OpenAI and Ollama)
- Added CHANGELOG.md file to document changes in each version of the package


### Fixed
- Fixed `parse_function`, so that it can be called within both `intra_pss`
and `inter_pss`
