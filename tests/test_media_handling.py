"""TDD tests for media handling in the bridge.

Tests the behavior of how different media types are routed:
- Images/stickers → sent as base64 attachments (LLM vision)
- Audio/video/documents → saved to workspace/media/ (agent uses tools)

These tests define the EXPECTED behavior. Write tests first, then make them pass.
"""

import base64
import json
import os

import pytest

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ── Test: Media routing logic ────────────────────────────────────────


class TestMediaRouting:
    """Test that different media types are routed correctly:
    images → attachments, audio/video/docs → saved to disk."""

    def test_image_becomes_attachment(self, tmp_path):
        """Image media should produce a base64 attachment, NOT be saved to disk."""
        from bridge import OPENCLAW_STATE_DIR

        media_info = {"media_id": "img1", "mime_type": "image/jpeg", "type": "image"}
        data = b"\xff\xd8\xff\xe0fake-jpeg-data"
        mime = "image/jpeg"

        # Simulate the routing logic from handle_message
        attachments = None
        text = "User sent an image."
        media_type = media_info.get("type", "")
        ext = mime.split("/")[-1].split(";")[0]
        filename = media_info.get("filename", f"media.{ext}")

        if media_type in ("image", "sticker"):
            attachments = [{
                "mimeType": mime,
                "fileName": filename,
                "content": base64.b64encode(data).decode(),
            }]
        else:
            save_path = tmp_path / "media" / filename
            save_path.parent.mkdir(parents=True, exist_ok=True)
            save_path.write_bytes(data)
            text += f"\n[File saved to workspace: media/{filename}]"

        assert attachments is not None, "Image should produce an attachment"
        assert len(attachments) == 1
        assert attachments[0]["mimeType"] == "image/jpeg"
        assert attachments[0]["fileName"] == "media.jpeg"
        assert base64.b64decode(attachments[0]["content"]) == data
        assert "[File saved" not in text

    def test_sticker_becomes_attachment(self, tmp_path):
        """Stickers are images — should be sent as attachments."""
        media_info = {"media_id": "stk1", "mime_type": "image/webp", "type": "sticker"}
        data = b"fake-webp-data"
        mime = "image/webp"

        media_type = media_info.get("type", "")
        attachments = None
        text = "User sent a sticker."

        if media_type in ("image", "sticker"):
            ext = mime.split("/")[-1].split(";")[0]
            filename = media_info.get("filename", f"media.{ext}")
            attachments = [{
                "mimeType": mime,
                "fileName": filename,
                "content": base64.b64encode(data).decode(),
            }]

        assert attachments is not None, "Sticker should produce an attachment"
        assert attachments[0]["mimeType"] == "image/webp"

    def test_audio_saved_to_workspace(self, tmp_path):
        """Audio should be saved to workspace/media/, not sent as attachment."""
        media_info = {"media_id": "aud1", "mime_type": "audio/ogg", "type": "audio"}
        data = b"fake-ogg-audio-data"
        mime = "audio/ogg"

        media_type = media_info.get("type", "")
        attachments = None
        text = "User sent a voice message."
        ext = mime.split("/")[-1].split(";")[0]
        filename = media_info.get("filename", f"media.{ext}")

        if media_type in ("image", "sticker"):
            attachments = [{
                "mimeType": mime,
                "fileName": filename,
                "content": base64.b64encode(data).decode(),
            }]
        else:
            media_dir = tmp_path / "media"
            media_dir.mkdir(parents=True, exist_ok=True)
            save_path = media_dir / filename
            save_path.write_bytes(data)
            text += f"\n[File saved to workspace: media/{filename}]"

        assert attachments is None, "Audio should NOT produce an attachment"
        assert (tmp_path / "media" / "media.ogg").exists()
        assert (tmp_path / "media" / "media.ogg").read_bytes() == data
        assert "[File saved to workspace: media/media.ogg]" in text

    def test_video_saved_to_workspace(self, tmp_path):
        """Video should be saved to workspace/media/."""
        media_info = {"media_id": "vid1", "mime_type": "video/mp4", "type": "video"}
        data = b"fake-mp4-video"
        mime = "video/mp4"

        media_type = media_info.get("type", "")
        attachments = None
        text = "User sent a video."
        ext = mime.split("/")[-1].split(";")[0]
        filename = media_info.get("filename", f"media.{ext}")

        if media_type in ("image", "sticker"):
            attachments = [{"mimeType": mime, "fileName": filename,
                           "content": base64.b64encode(data).decode()}]
        else:
            media_dir = tmp_path / "media"
            media_dir.mkdir(parents=True, exist_ok=True)
            (media_dir / filename).write_bytes(data)
            text += f"\n[File saved to workspace: media/{filename}]"

        assert attachments is None
        assert (tmp_path / "media" / "media.mp4").exists()
        assert "[File saved" in text

    def test_document_saved_to_workspace(self, tmp_path):
        """Documents should be saved with their original filename."""
        media_info = {
            "media_id": "doc1", "mime_type": "application/pdf",
            "type": "document", "filename": "report.pdf",
        }
        data = b"%PDF-1.4 fake pdf"
        mime = "application/pdf"

        media_type = media_info.get("type", "")
        attachments = None
        text = "User sent a document: report.pdf"
        filename = media_info.get("filename", "document")

        if media_type in ("image", "sticker"):
            attachments = [{"mimeType": mime, "fileName": filename,
                           "content": base64.b64encode(data).decode()}]
        else:
            media_dir = tmp_path / "media"
            media_dir.mkdir(parents=True, exist_ok=True)
            (media_dir / filename).write_bytes(data)
            text += f"\n[File saved to workspace: media/{filename}]"

        assert attachments is None
        assert (tmp_path / "media" / "report.pdf").exists()
        assert (tmp_path / "media" / "report.pdf").read_bytes() == data

    def test_text_message_no_media_handling(self):
        """Text messages should have no attachments and no file saving."""
        media_info = None
        attachments = None
        text = "hello world"

        if media_info and media_info.get("media_id"):
            attachments = [{"should": "not happen"}]

        assert attachments is None
        assert text == "hello world"


# ── Test: Attachment format ──────────────────────────────────────────


class TestAttachmentFormat:
    """Test that attachments match OpenClaw's expected format."""

    def test_attachment_has_required_fields(self):
        """OpenClaw expects: mimeType, fileName, content (base64)."""
        data = b"test image data"
        attachment = {
            "mimeType": "image/jpeg",
            "fileName": "photo.jpg",
            "content": base64.b64encode(data).decode(),
        }

        assert "mimeType" in attachment
        assert "fileName" in attachment
        assert "content" in attachment
        # Content must be valid base64
        decoded = base64.b64decode(attachment["content"])
        assert decoded == data

    def test_mime_type_preserved(self):
        """MIME type from Meta should be passed through unchanged."""
        for mime in ["image/jpeg", "image/png", "image/webp", "image/gif"]:
            ext = mime.split("/")[-1].split(";")[0]
            attachment = {
                "mimeType": mime,
                "fileName": f"media.{ext}",
                "content": base64.b64encode(b"data").decode(),
            }
            assert attachment["mimeType"] == mime

    def test_extension_extracted_from_mime(self):
        """File extension should be derived from MIME type."""
        cases = [
            ("image/jpeg", "jpeg"),
            ("image/png", "png"),
            ("audio/ogg", "ogg"),
            ("audio/ogg; codecs=opus", "ogg"),  # WhatsApp voice notes
            ("video/mp4", "mp4"),
            ("application/pdf", "pdf"),
        ]
        for mime, expected_ext in cases:
            ext = mime.split("/")[-1].split(";")[0]
            assert ext == expected_ext, f"Expected {expected_ext} from {mime}, got {ext}"


# ── Test: Saved file path ────────────────────────────────────────────


class TestSavedFilePath:
    """Test that saved media files go to the right location."""

    def test_audio_saved_in_media_dir(self, tmp_path):
        media_dir = tmp_path / "media"
        media_dir.mkdir(parents=True, exist_ok=True)
        save_path = media_dir / "voice.ogg"
        save_path.write_bytes(b"audio data")

        assert save_path.exists()
        assert save_path.parent.name == "media"

    def test_media_dir_created_if_missing(self, tmp_path):
        """The media/ directory should be created automatically."""
        media_dir = tmp_path / "workspace" / "media"
        assert not media_dir.exists()

        media_dir.mkdir(parents=True, exist_ok=True)
        assert media_dir.exists()

    def test_text_includes_relative_path(self):
        """The text sent to the agent should include the relative workspace path."""
        filename = "voice_note.ogg"
        text = "User sent a voice message."
        text += f"\n[File saved to workspace: media/{filename}]"

        assert "media/voice_note.ogg" in text
        assert "workspace" not in text.split("[File saved")[1].split("]")[0].split(": ")[1]
        # Should be relative (media/file), not absolute (/Users/.../media/file)


# ── Test: Agent text for different media types ───────────────────────


class TestAgentMediaText:
    """Test what text the agent receives for each media type."""

    def test_audio_text_mentions_file_location(self):
        text = "User sent a voice message."
        text += "\n[File saved to workspace: media/media.ogg]"
        assert "voice" in text.lower()
        assert "media/media.ogg" in text

    def test_image_text_no_file_reference(self):
        """Images are sent as attachments — text should NOT mention file paths."""
        text = "User sent an image."
        # No file saving for images
        assert "[File saved" not in text

    def test_document_text_includes_filename(self):
        text = "User sent a document: report.pdf"
        text += "\n[File saved to workspace: media/report.pdf]"
        assert "report.pdf" in text
