import os
import cv2
from typing import Callable, Optional, Tuple

class VideoExtractor:
    """Handles video frame extraction with customizable intervals and resizing."""

    def __init__(self):
        self._is_cancelled = False

    def cancel(self):
        """Cancel ongoing extraction."""
        self._is_cancelled = True

    def extract(
        self,
        video_path: str,
        output_dir: str,
        mode: str = "frame",  # "frame" or "time"
        frame_interval: int = 15,
        time_interval_ms: int = 500,
        max_size: Optional[int] = None,
        prefix: str = "frame",
        progress_callback: Optional[Callable[[int, int, int], None]] = None,
    ) -> Tuple[bool, str, int]:
        """
        Extract frames from a video file.

        Args:
            video_path: Path to input video.
            output_dir: Directory where extracted frames will be saved.
            mode: "frame" (extract every N frames) or "time" (extract every X ms).
            frame_interval: Interval in frames if mode is "frame".
            time_interval_ms: Interval in milliseconds if mode is "time".
            max_size: Optional maximum width/height (aspect ratio preserved).
            prefix: Filename prefix for extracted frames.
            progress_callback: Callback func(current_frame, total_frames, saved_count).

        Returns:
            Tuple of (success, message, total_saved_count).
        """
        self._is_cancelled = False

        if not os.path.isfile(video_path):
            return False, f"File video tidak ditemukan: {video_path}", 0

        os.makedirs(output_dir, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return False, f"Gagal membuka video: {video_path}", 0

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0:
            fps = 30.0  # Fallback assumption

        # Calculate frame step
        if mode == "time":
            step = max(1, int(round((time_interval_ms / 1000.0) * fps)))
        else:
            step = max(1, int(frame_interval))

        current_frame_idx = 0
        saved_count = 0

        try:
            while True:
                if self._is_cancelled:
                    cap.release()
                    return False, "Ekstraksi dibatalkan oleh pengguna.", saved_count

                ret, frame = cap.read()
                if not ret:
                    break

                if current_frame_idx % step == 0:
                    saved_count += 1
                    # Resize if requested
                    if max_size and max_size > 0:
                        h, w = frame.shape[:2]
                        if max(h, w) > max_size:
                            scale = max_size / float(max(h, w))
                            new_w = int(w * scale)
                            new_h = int(h * scale)
                            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

                    filename = f"{prefix}_{saved_count:05d}.jpg"
                    out_path = os.path.join(output_dir, filename)
                    cv2.imwrite(out_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

                current_frame_idx += 1

                if progress_callback and current_frame_idx % 5 == 0:
                    progress_callback(current_frame_idx, total_frames, saved_count)

        except Exception as e:
            cap.release()
            return False, f"Error saat mengekstrak video: {str(e)}", saved_count

        cap.release()
        if progress_callback:
            progress_callback(total_frames, total_frames, saved_count)

        return True, f"Berhasil mengekstrak {saved_count} frame.", saved_count

    def extract_batch(
        self,
        video_paths: list,
        output_dir: str,
        create_subfolders: bool = True,
        mode: str = "frame",
        frame_interval: int = 15,
        time_interval_ms: int = 500,
        max_size: Optional[int] = None,
        prefix: str = "amr_frame",
        progress_callback: Optional[Callable[[int, int, str, int, int, int], None]] = None,
    ) -> Tuple[bool, str, int]:
        """
        Extract frames from multiple video files sequentially.

        Args:
            video_paths: List of paths to input video files.
            output_dir: Root directory where frames will be saved.
            create_subfolders: If True, creates a subfolder per video (e.g. output_dir/<video_name>/).
            mode: "frame" or "time".
            frame_interval: Interval in frames if mode is "frame".
            time_interval_ms: Interval in milliseconds if mode is "time".
            max_size: Optional maximum dimension for resizing.
            prefix: Filename prefix for extracted frames.
            progress_callback: Callback func(current_video_idx, total_videos, video_name, current_frame, total_frames, total_saved_count).

        Returns:
            Tuple of (success, message, total_saved_count).
        """
        self._is_cancelled = False

        if not video_paths:
            return False, "Tidak ada file video yang dipilih.", 0

        valid_videos = [p for p in video_paths if os.path.isfile(p)]
        if not valid_videos:
            return False, "Tidak ada file video valid yang ditemukan.", 0

        os.makedirs(output_dir, exist_ok=True)
        total_videos = len(valid_videos)
        total_saved_count = 0
        success_videos = 0
        failed_videos = []

        for v_idx, v_path in enumerate(valid_videos, start=1):
            if self._is_cancelled:
                return False, f"Ekstraksi dibatalkan. {total_saved_count} frame tersimpan dari {success_videos} video.", total_saved_count

            v_name = os.path.splitext(os.path.basename(v_path))[0]
            # Sanitize folder name
            safe_name = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in v_name).strip("_") or f"video_{v_idx}"

            if create_subfolders:
                target_dir = os.path.join(output_dir, safe_name)
                os.makedirs(target_dir, exist_ok=True)
                item_prefix = prefix or "frame"
            else:
                target_dir = output_dir
                item_prefix = f"{prefix}_{safe_name}" if prefix else safe_name

            def single_progress(cur_f, tot_f, cur_saved):
                if progress_callback:
                    progress_callback(v_idx, total_videos, os.path.basename(v_path), cur_f, tot_f, total_saved_count + cur_saved)

            ok, msg, saved = self.extract(
                video_path=v_path,
                output_dir=target_dir,
                mode=mode,
                frame_interval=frame_interval,
                time_interval_ms=time_interval_ms,
                max_size=max_size,
                prefix=item_prefix,
                progress_callback=single_progress,
            )

            total_saved_count += saved
            if ok:
                success_videos += 1
            else:
                if self._is_cancelled:
                    return False, f"Ekstraksi dibatalkan. {total_saved_count} frame tersimpan dari {success_videos} video.", total_saved_count
                failed_videos.append(os.path.basename(v_path))

        if failed_videos:
            msg = f"Selesai dengan catatan: {total_saved_count} frame dari {success_videos}/{total_videos} video berhasil diekstrak. Gagal: {', '.join(failed_videos)}"
            return True, msg, total_saved_count

        msg = f"Berhasil mengekstrak {total_saved_count} frame dari seluruh {total_videos} video."
        return True, msg, total_saved_count
