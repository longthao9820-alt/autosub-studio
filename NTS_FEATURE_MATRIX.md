# Đối chiếu NTS AutoSub 9.2 → AutoSub Studio V2

Báo cáo dựa trên kiểm kê read-only thư mục NTS, SQLite `db/app.db`, 20 project thật,
17 cơ sở dữ liệu OCR raw, project `220-46`, draft xuất mẫu thật và payload Nuitka đã giải nén.
Không sao chép dữ liệu đăng nhập, cookie, token, proxy hay khóa API.

## Đã có tương đương dùng được

| Nhóm NTS | AutoSub Studio V2 |
|---|---|
| Danh sách/tạo/xóa/mở project | Có, kèm hàng đợi nhiều video và phục hồi tác vụ gián đoạn |
| Nhập project NTS | Đọc trực tiếp `db/app.db`, chọn project, nhập video/sub/time/dịch/người nói/cấu hình |
| OCR hardsub | PP-OCRv4 Mobile, GPU, vùng crop, màu/height/filter, 15 fps, cache SQLite từng frame |
| ASR | Faster-Whisper local CPU/GPU, VAD, giới hạn dòng/thời lượng |
| Sửa sub | Bảng gốc/dịch, time, tách/gộp/thêm/xóa, undo/redo, kiểm tra lỗi |
| Dịch | Google miễn phí / AI Gateway (chuẩn OpenAI HTTP, model `sub`/`prime`, thinking) / không dịch, batch, glossary, cache |
| Giọng đọc offline | Piper Local (ONNX, đa ngôn ngữ, hoàn toàn offline, tải và quản lý model độc lập) |
| Danh sách giọng Nam/Nữ | Có profile mặc định/Nam/Nữ |
| Chỉnh voice | Volume, speed, Bass/Mid/Treble, từ điển phát âm, dấu câu/khoảng nghỉ |
| CANH THEO TIME SUB | Giữ nguyên timeline gốc, xếp voice theo mốc time sub, tốc độ chung cố định, chống chồng hoặc cho phép chồng |
| CANH THEO GIỌNG ĐỌC | Kéo từng đoạn video tới `max(khe gốc, voice)`, render song song, ghép lại |
| Lưu/cache voice | Có cache dùng chung và tùy chọn lưu WAV từng câu trong project |
| Giữ tiếng gốc | Có volume, ducking, nhạc tách riêng hoặc audio trực tiếp từ video |
| Xuất video lồng tiếng | MP4 với video copy + AAC, không còn chỉ tạo WAV |
| Render subtitle | ASS, blur sub gốc, LUT, GPU encoder, 135 font và 48 LUT từ NTS |
| Xuất video / render | RenderPanel trực tiếp (MP4 H.264/AAC, phụ đề cứng ASS, blur) và xuất gói dự án |
| Tải video | `yt-dlp` portable, tiến trình, hủy, MP4 merge |
| Cấu hình tùy chỉnh | Tạo/chọn/xóa profile thật, lưu toàn bộ tham số workflow |
| Import/export project | JSON package và importer NTS SQLite |
| Kiểm kê NTS | `scripts/audit_nts.py`, báo cáo `NTS_AUDIT.json`, tự che trường nhạy cảm |

## Tương đương bằng engine khác

| NTS | AutoSub Studio |
|---|---|
| VapourSynth BestSource timeline | FFmpeg segment retime + concat; kiểm thử H.264/AAC thành công |
| Voice clone `tts_clone_*` từ server NTS | Danh mục Local Voice (Piper Local) chất lượng cao tải về máy; không giả mạo tên/model clone |
| Audio separator AI rất lớn trong `ffsub_v2` | FFmpeg local; dùng Demucs nếu môi trường có cài |
| PP-OCRv5 server trong `ffsub_v2` | PP-OCRv4 Mobile nhanh và PP-OCRv6 Small/Medium có sẵn |

## Phụ thuộc dịch vụ riêng của NTS, không thể phục dựng chỉ từ thư mục local

- Đăng nhập, gia hạn, phân quyền/quản lý nhân viên.
- Máy chủ giọng clone riêng và số ký tự còn lại của tài khoản NTS.
- Endpoint/API private cho Minimax, ElevenLabs hoặc model dịch private nếu không có khóa hợp lệ.
- Chức năng cập nhật qua `update.exe` và dịch vụ bản quyền phía máy chủ.

Các mục trên cần đặc tả API/server và thông tin truy cập do chủ sở hữu cung cấp. Phần mềm
không trích xuất hoặc nhúng lại credential từ NTS.
