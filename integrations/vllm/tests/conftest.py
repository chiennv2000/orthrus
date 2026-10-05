# SPDX-License-Identifier: Apache-2.0
"""Ensure the plugin's registrations are active for tests run standalone
(outside a real vLLM engine, which would otherwise call this via its
plugin-loading mechanism)."""

import orthrus_vllm

orthrus_vllm.register()
