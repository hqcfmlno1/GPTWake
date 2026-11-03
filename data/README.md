# Local datasets

Dataset contents are ignored by Git. Preserve the original audio and metadata.
Future manifests should include `audio_path`, `label`, `speaker_id`,
`source_type`, `accent`, `transcript`, `sample_rate`, `duration`,
`recording_session_id`, `original_recording_id`, and `split`, with phrase start
and end timestamps where available.

Split speakers and recording groups before creating augmented variants or
windows. A positive recording does not imply every crop is positive: the crop
must contain the complete valid phrase. Specify a policy for partial phrases.

Do not put API keys, credentials, or tokens in manifests.
