# AutoSub Studio V2

Phần mềm Windows để **tách phụ đề, dịch phụ đề, lồng tiếng offline và render video** — chạy trực tiếp trên
máy của bạn, bảo đảm an toàn dữ liệu và quyền riêng tư.

---

## Điểm mới trong phiên bản 2.2.0

- **Nhận dạng Vision AI thực thụ (True AI Recognition):** Tách phụ đề video bằng role `sub` qua AI Gateway, tuyệt đối không fallback âm thầm về local OCR; role `prime` dành riêng cho dịch phụ đề.
- **Chỉ xử lý vùng cắt (Crop-only):** Cắt chính xác vùng phụ đề (`region crop`), chỉ gửi ảnh vùng phụ đề lên AI, loại bỏ chữ gây nhiễu bên ngoài, bảo mật và tiết kiệm băng thông.
- **Bộ lọc thị giác (Visual Filter):** Tự động lọc bỏ khung hình tĩnh và trùng lặp trước khi gửi, giảm số lượng request thực tế trên 70%.
- **Điều phối lô toàn cục (Batch & Global Scheduler):** Gom nhóm frame theo lô (`batch_size`), kiểm soát số luồng đồng thời toàn cục (`max_concurrency`), chống tràn hàng đợi.
- **Thử lại, lưu điểm ngắt & khôi phục (Retries / Checkpoint / Resume):** Cơ chế thử lại lỗi tạm thời; lưu tiến độ theo từng batch vào SQLite checkpoint, cho phép tiếp tục tác vụ khi bị ngắt.
- **Giao diện đáp ứng (Responsive UI):** Thiết kế lại panel B1 và khối cấu hình AI theo lưới `QGridLayout`, tự động gói gọn không bị cắt mép trên màn hình 1366×768.
- *(Lệnh đo kiểm hiệu năng AI OCR mô phỏng đã có sẵn tại Mục 8).*

---

## 1. Mở phần mềm

**Cách nhanh nhất:** vào thư mục `dist\AutoSubStudio\` rồi **nhấp đúp vào `AutoSubStudio.exe`**.

Lần đầu mở, Windows có thể hiện bảng "Windows protected your PC" vì phần mềm chưa có chữ ký
số. Bấm **More info** → **Run anyway**.

Muốn mở nhanh hơn về sau: nhấp chuột phải vào `AutoSubStudio.exe` → **Send to** →
**Desktop (create shortcut)**.

**Không cần cài đặt phức tạp.** Bản đóng gói chính thức đi kèm sẵn FFmpeg, mô hình đọc chữ trên hình (PP-OCR), runtime giọng đọc Piper Local (`_internal\piper\`), và các thư viện tăng tốc GPU. Mô hình nhận dạng giọng nói ASR (`faster-whisper`) được tách khỏi bộ cài để đảm bảo dung lượng phát hành <2 GiB; mô hình được tự động tải về thư mục `Data\models\asr\` khi dùng lần đầu, hoặc tự động chuyển an toàn từ bản V1 cũ sang.

---

## 1a. Tăng tốc bằng card đồ họa

Phần mềm **tự phát hiện và dùng card đồ họa** của máy, không cần bật thủ công (số liệu tham khảo trước đây trên card NVIDIA RTX 3060):

| Việc | Bằng CPU | Bằng card đồ họa | Ghi chú |
|---|---|---|---|
| Nhận dạng giọng nói (Whisper) | Chạy ổn định | Nhanh gấp ~3,6 lần (tham khảo) | Cần card NVIDIA + driver CUDA |
| Đọc chữ trên hình (OCR) | Chạy ổn định | Nhanh gấp ~4,7 lần (tham khảo) | Cần onnxruntime-gpu tương thích |
| Render video & phụ đề cứng | Chạy ổn định | Nhanh gấp ~4,0 lần (tham khảo) | Hỗ trợ NVIDIA NVENC, Intel Quick Sync, AMD AMF |

Máy nào dùng được gì:

| Loại card | Nhận dạng giọng nói | Đọc chữ (OCR) | Render video |
|---|---|---|---|
| NVIDIA (có driver) | ✅ Tăng tốc CUDA | ✅ Tăng tốc ONNX CUDA | ✅ NVENC |
| Intel (chip tích hợp) | ❌ CPU | ❌ CPU | ✅ Quick Sync |
| AMD | ❌ CPU | ❌ CPU | ✅ AMF |
| Không có card rời | ❌ CPU | ❌ CPU | ❌ CPU (libx264) |

Nếu máy không có card phù hợp, phần mềm **tự chuyển sang CPU** an toàn, không báo lỗi ngắt quãng.
Muốn tắt tăng tốc GPU: vào tab **Cấu Hình Chung** (Cài đặt chung), bỏ chọn các mục tăng tốc phần cứng.

---

## 1b. Mang sang máy khác

Chép **nguyên cả thư mục** `dist\AutoSubStudio` sang máy Windows khác (qua USB hoặc ổ cứng ngoài),
rồi nhấp đúp `AutoSubStudio.exe` là chạy ngay.

Dự án và cấu hình nằm trọn vẹn trong thư mục `Data` bên trong thư mục phần mềm (chế độ di động portable).

Có hai tệp kiểm tra môi trường trong thư mục:

| Tệp | Mục đích | Thời gian |
|---|---|---|
| `KIEM-TRA-MAY-NAY.bat` | Kiểm tra nhanh FFmpeg, mô hình, runtime, quyền ghi đĩa | ~10 giây |
| `KIEM-TRA-SAU.bat` | Chạy thử toàn diện: OCR, giọng đọc Piper, nhận dạng giọng nói, render video | 1–2 phút |

---

## 2. Cấu hình AI Gateway

AI Gateway cho phép kết nối các mô hình AI ngôn ngữ lớn theo chuẩn OpenAI HTTP API (`/chat/completions`):

1. Vào tab **Cấu Hình Chung** → bấm nút **AI Gateway ⚙** (trong nhóm cấu hình dịch và AI).
2. Nhập **Endpoint** (ví dụ URL gateway tương thích chuẩn OpenAI) và **Khóa API**. Khóa được mã hóa an toàn theo tài khoản Windows trong `Data\config\secrets.dat`.
3. Cấu hình role model do Gateway định tuyến:
   - `sub`: dành riêng cho trích xuất chữ phụ đề từ hình/video.
   - `prime`: dành riêng cho dịch nội dung phụ đề/SRT.
   - Người dùng không cần chọn role ở từng tác vụ: **Lấy Sub** tự dùng `sub`, **Dịch Sub** tự dùng `prime`.
   - Mức độ suy luận (*Thinking level*): tắt (`off`), thấp (`low`), vừa (`medium`), cao (`high`).
4. Bấm **Kiểm tra kết nối** để kiểm tra trực tiếp endpoint và danh sách model khả dụng.

---

## 3. Luồng làm việc 4 bước (B1 → B4)

### Bước 1: Tách phụ đề (B1)

- **Tách bằng giọng nói (ASR):** Sử dụng Faster-Whisper chạy trực tiếp trên máy (CPU hoặc GPU). Mô hình mặc định là `small`, lưu bền vững trong `Data\models\asr\`. Lần đầu sử dụng sẽ tự động tải về; các lần sau hoặc khi mở lại/offline sẽ chạy trực tiếp từ đĩa không cần mạng.
- **Tách bằng chữ trên hình (OCR):**
  - Dành cho video có phụ đề cứng (hardsub).
  - Vùng nhận dạng trực quan tại màn hình *Screen Edit* với khung chữ nhật điều chỉnh linh hoạt.
  - Sử dụng engine PP-OCRv6 Medium hoặc PP-OCRv4 Mobile, hỗ trợ lọc màu chữ (`#RRGGBB`) và lọc chiều cao chữ để loại bỏ logo/quảng cáo.
  - Khi chọn AI Gateway, local chỉ xác định vùng và thời điểm thay đổi; role `sub` đọc chữ từ các frame đại diện theo lô. Không fallback âm thầm sang PP-OCR.
  - Chế độ quét mốc thời gian 2 lần giúp mốc khớp chính xác với khung hình xuất hiện/biến mất của phụ đề.
  - Tự động xuất tệp phụ đề `.srt` cùng tên bên cạnh tệp video gốc.

### Bước 2: Dịch nội dung (B2)

Thao tác dịch xử lý trực tiếp trên danh sách câu phụ đề, không làm thay đổi timeline video:

- **Lựa chọn dịch:**
  - `Google (miễn phí)`: dịch nhanh qua Google Dịch (cần Internet).
  - `AI Gateway` / `Server AI API`: tự dùng role `prime`, ưu tiên gửi toàn bộ SRT trong một structured task; file lớn được chia theo kích thước và chỉ sửa lại ID/chunk lỗi. Timestamp gốc không đổi.
  - `Không dịch (giữ nguyên)`: giữ nguyên nội dung gốc để biên tập thủ công.
- **Bảng thuật ngữ (Glossary):** cố định cách dịch tên riêng, địa danh, thuật ngữ chuyên ngành.
- **Kiểm tra lỗi phụ đề:** bấm **Kiểm tra time** để phát hiện các câu bị trùng lấn thời gian, thời lượng quá ngắn hoặc khoảng trống bất thường.

### Bước 3: Thư viện giọng đọc & Ghép giọng (B3)

Hệ thống giọng đọc V2 sử dụng **Local Voice (Piper Local)** chạy hoàn toàn offline trên máy tính:

- **Thư viện giọng đọc (Catalog):**
  - Cung cấp danh mục giọng đọc chất lượng cao đa ngôn ngữ: Tiếng Việt (`vi_VN-vais1000-medium`), Tiếng Anh (`en_US-bryce-medium`, `en_US-libritts_r-medium`), Tiếng Trung (`zh_CN-chaowen-medium`), v.v.
  - Trạng thái rõ ràng: *Đã tải*, *Đang tải*, hoặc *Chưa tải*.
  - Người dùng chủ động tải mô hình mong muốn bằng nút **Tải Voice**. Quá trình tải kiểm tra mã băm SHA256 nguyên tử; có thể xóa mô hình để giải phóng dung lượng đĩa khi cần.
  - Runtime Piper (`piper.exe`) được nhúng trong bản đóng gói; khi chạy từ mã nguồn, cần đặt vào `assets/piper/`.
- **Chế độ canh thời gian (Voice Timing Mode):**
  - *Canh theo phụ đề (Subtitle Timing):* Giữ nguyên timeline video gốc. Mỗi câu đọc bắt đầu chính xác tại mốc thời gian bắt đầu của câu phụ đề tương ứng. Tốc độ đọc toàn cục (global speed) được giữ cố định theo thiết lập, không tự động co giãn hay thay đổi tốc độ riêng theo từng câu. Nếu câu đọc dài hơn thời lượng phụ đề, âm thanh có thể gối đầu (chồng) sang câu kế tiếp nếu bật tùy chọn cho phép chồng tiếng, hoặc được mix liền mạch mà không làm biến dạng thời lượng video gốc.
  - *Canh theo giọng đọc (Voice Timing):* Tự động phân đoạn và kéo giãn timeline video theo độ dài phát âm thực tế của từng câu voice (`max(khe gốc, voice)`), đảm bảo video khớp trọn vẹn với lời thoại tự nhiên.
- **Tùy chỉnh âm thanh:**
  - Tốc độ đọc (speed / length_scale), âm lượng, bộ cân bằng âm thanh Bass/Mid/Treble, khoảng nghỉ cuối câu.
  - Từ điển phát âm tùy chỉnh (`từ_gốc=cách_đọc`).
  - Phân tách và gán giọng theo nhân vật hoặc giới tính (Nam / Nữ / Mặc định).
  - Tự động ducking: hạ âm lượng âm thanh gốc (mặc định xuống 20%) khi giọng lồng tiếng vang lên.
  - Xuất thành video lồng tiếng hoàn chỉnh (`*_long_tieng.mp4`) hoặc xuất riêng tệp âm thanh WAV.

### Bước 4: Render & Xuất video (B4)

Thay thế các luồng xuất bên thứ ba cũ bằng **RenderPanel** tích hợp sẵn:

- **Render phụ đề cứng (Hardsub):** gắn trực tiếp phụ đề vào khung hình video bằng định dạng ASS chuyên nghiệp (tùy chỉnh phông chữ, cỡ chữ, màu sắc, viền, bóng đổ, vị trí lề).
- **Bộ lọc hình ảnh:**
  - Che mờ phụ đề cũ (Blur sub gốc) theo vùng chọn.
  - Áp dụng bảng màu điện ảnh (hơn 48 LUT đi kèm).
- **Mã hóa video hiệu năng cao:** xuất định dạng chuẩn MP4 (H.264 / AAC). Tự động lựa chọn phần cứng GPU nhanh nhất (NVENC, Quick Sync, AMF) hoặc CPU đa luồng với thiết lập CRF và preset tối ưu.

---

## 4. Quản lý cấu hình, đầu ra và cập nhật

### Phân biệt Presets kết xuất và Profiles cấu hình chung

- **Cấu hình Screen Render (Render Presets):**
  - Quản lý các mẫu tham số kết xuất video riêng cho tab *B4: Render & Xuất Video* (độ phân giải, tốc độ khung hình fps, CRF, bitrate, định dạng kiểu chữ ASS, lề phụ đề).
  - Lưu trữ độc lập trong tệp `Data\config\render_presets.json`.
  - Cho phép người dùng lưu và chọn nhanh các định dạng chuẩn (video ngang 16:9, video dọc 9:16, mẫu đăng tải TikTok, YouTube).
- **Cấu hình chung toàn bộ phần mềm (General Config Profiles):**
  - Lưu trữ toàn bộ trạng thái và tham số workflow (mô hình ASR, nhà cung cấp dịch, thông số lồng tiếng, ducking, OCR, v.v.) trong `Data\config\config.json` dưới mục `config_profiles`.
  - Được tạo mới, lưu, áp dụng hoặc xóa tại tab *Cấu Hình Chung* với profile mặc định ban đầu là `DEFAULT`.

### Thư mục xuất đầu ra (Output Folder)

- Khi người dùng thiết lập thư mục tại tab *Cấu Hình Chung* (hoặc nút *Thư Mục Xuất...* trong tab B4), phần mềm chỉ sao chép **tệp video render hoàn thiện cuối cùng** vào thư mục này sau khi quá trình render hoàn tất thành công.
- Tệp phụ đề `.srt` và video lồng tiếng nháp vẫn được quản lý trong thư mục dự án (`exports/`) hoặc bên cạnh video gốc theo đúng thiết lập quy trình.

### Tự động kiểm tra cập nhật (Self-Updater)

Tại tab *Cấu Hình Chung* (nhóm Cập Nhật Phần Mềm):

- **Tự động kiểm tra bản cập nhật khi khởi động:** tùy chọn bật/tắt kiểm tra phiên bản mới mỗi khi mở ứng dụng.
- **Kiểm Tra Cập Nhật:** gửi yêu cầu kiểm tra phiên bản mới nhất từ GitHub Releases của dự án. Nếu có bản mới hơn, thông tin cập nhật sẽ hiển thị và kích hoạt nút **Cập Nhật Ngay**. Nếu đang ở bản mới nhất hoặc không có kết nối mạng, phần mềm sẽ hiển thị thông báo trạng thái tương ứng.
- **Cập Nhật Ngay:** tải gói cập nhật về thư mục tạm, khởi chạy trình hỗ trợ `AutoSubUpdater.exe`, đóng ứng dụng chính để thực hiện thay thế tệp nguyên tử và tự động khởi động lại ứng dụng sau khi cập nhật thành công.
- *Lưu ý hiện tại:* Nguồn cập nhật theo dõi qua GitHub Releases của dự án; hiện tại chưa có bản phát hành V2 chính thức trên máy chủ từ xa, nên việc kiểm tra sẽ xác nhận ứng dụng đang ở phiên bản mới nhất.

---

## 5. Dữ liệu lưu ở đâu

Bản đóng gói hoạt động ở chế độ di động (Portable), toàn bộ dữ liệu nằm trong thư mục `Data/`:

| Thư mục / Tệp | Nội dung |
|---|---|
| `Data\workspace\projects\` | Các dự án, video, âm thanh, phụ đề, sản phẩm xuất |
| `Data\workspace\db\app.db` | Cơ sở dữ liệu SQLite quản lý danh sách dự án và kịch bản |
| `Data\config\config.json` | Cấu hình tham số và các profile cấu hình chung |
| `Data\config\render_presets.json` | Các mẫu cấu hình kết xuất video (Screen Render Presets) |
| `Data\config\secrets.dat` | Khóa API được mã hóa an toàn theo tài khoản Windows |
| `Data\models\asr\` | Các mô hình nhận dạng giọng nói (Whisper) đã tải hoặc chuyển từ bản cũ |
| `Data\models\piper\` | Các mô hình giọng đọc offline đã tải |
| `Data\config\logs\crash.log` | Nhật ký sự cố khi xảy ra lỗi ngoài ý muốn |
| `_internal\` | FFmpeg, runtime Piper, mô hình OCR và thư viện GPU |

Khi chạy từ mã nguồn lập trình (không dùng bản đóng gói), dữ liệu lưu tại thư mục người dùng:
`Documents\AutoSubStudio\` và `AppData\Roaming\AutoSubStudio\`.

---

## 6. Yêu cầu kết nối mạng

- **Không cần Internet (chạy hoàn toàn offline):**
  - Nhận dạng giọng nói (Whisper) khi mô hình đã có trong `Data\models\asr\`.
  - Đọc chữ trên hình ảnh (OCR).
  - Tách nhạc nền cơ bản.
  - Lồng tiếng với Local Voice (Piper Local) khi mô hình đã có trên máy.
  - Render và xuất video phụ đề cứng.
- **Cần kết nối Internet:**
  - Tải mô hình nhận dạng giọng nói (Whisper) lần đầu tiên (tự động lưu vào `Data\models\asr\`).
  - Dịch phụ đề bằng Google Dịch hoặc AI Gateway.
  - Tải mô hình giọng đọc Piper về máy lần đầu tiên.
  - Kiểm tra cập nhật phiên bản mới.

---

## 7. Đóng gói bản phát hành (Build & Release)

Script đóng gói `scripts\build.ps1` chuẩn bị và đóng gói ứng dụng di động hoàn chỉnh:

Thành phần cần chuẩn bị trong thư mục `assets/` trước khi đóng gói:
```
assets\ffmpeg\ffmpeg.exe
assets\ffmpeg\ffprobe.exe
assets\piper\piper.exe            # Tải từ https://github.com/rhasspy/piper/releases
assets\piper\espeak-ng-data\      # Đi kèm trong gói Piper Windows release
assets\ocr\*.onnx                 # Mô hình PP-OCRv4
assets\cuda\*.dll
```
*(Lưu ý: Mô hình ASR không được nhúng vào bản đóng gói để giữ gói phát hành GitHub Release <2 GiB; người dùng tải về qua `Data\models\asr\` khi dùng).*

Lệnh thực hiện đóng gói trong PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\build.ps1
```

Sản phẩm hoàn chỉnh sẽ được tạo ra tại thư mục `dist\AutoSubStudio\`, sẵn sàng sử dụng ngay bằng cách nhấp đúp `AutoSubStudio.exe`.

---

## 8. Dành cho nhà phát triển

```powershell
# Kích hoạt môi trường và chạy ứng dụng từ mã nguồn
.\.venv\Scripts\python.exe -m autosub_studio

# Chạy toàn bộ bộ kiểm thử tự động
.\.venv\Scripts\python.exe -m pytest

# Kiểm tra chất lượng và định dạng mã nguồn
.\.venv\Scripts\python.exe -m ruff check .

# Kiểm tra kiểu dữ liệu tĩnh
.\.venv\Scripts\python.exe -m mypy

# Kiểm tra tự động các luồng chức năng
.\.venv\Scripts\python.exe scripts\smoke_media.py
.\.venv\Scripts\python.exe scripts\smoke_speech.py
.\.venv\Scripts\python.exe -m autosub_studio --selftest-full

# Đo hiệu năng AI OCR mô phỏng (không cần mạng, không tốn API)
.\.venv\Scripts\python.exe scripts\benchmark_ai_ocr.py --profile all --json benchmark-results\ai_ocr_latest.json
```

### Đo hiệu năng AI OCR (AI OCR Benchmark)

Hệ thống cung cấp công cụ đo kiểm hiệu năng lặp lại được (`scripts\benchmark_ai_ocr.py`), chạy hoàn toàn nội bộ bằng dữ liệu khung hình thị giác mô phỏng (synthetic visual manifests), **không gọi mạng bên ngoài và không tốn chi phí API**.

Kết quả benchmark Gateway thực tế của đợt refactor được ghi tại [AI_GATEWAY_REFACTOR_BENCHMARK.md](AI_GATEWAY_REFACTOR_BENCHMARK.md).

Lệnh thực thi:

```powershell
.\.venv\Scripts\python.exe scripts\benchmark_ai_ocr.py --profile all --json benchmark-results\ai_ocr_latest.json
```

**Ý nghĩa các chỉ số trong bảng kết quả:**

| Cột | Ý nghĩa |
|---|---|
| `Profile` | Hồ sơ đo: `short` (120 khung ~1 phút), `normal` (1.800 khung ~15 phút), `long` (7.200 khung ~60 phút), `multi` (3 luồng video chạy đồng thời). |
| `Sampled` | Tổng số khung hình gốc được lấy mẫu theo fps. |
| `Sent` | Số khung hình đại diện thực tế gửi lên AI sau khi đã lọc bỏ khung trống và khung trùng lặp. |
| `Reqs` | Số lượng yêu cầu theo lô (`ceil(Sent / batch_size)`). |
| `AvgBatch` | Kích thước trung bình của mỗi lô gửi (số ảnh/lô). |
| `Reduction` | Tỷ lệ giảm tải khung hình (`(Sampled - Sent) / Sampled`), luôn đảm bảo đạt `>= 70%`. |
| `Subs` | Số câu phụ đề trích xuất, đảm bảo 100% khớp dữ liệu chuẩn (Ground Truth) và 0 ảo giác (hallucinations). |
| `Latency` | Độ trễ xử lý trung bình mỗi lô yêu cầu (mô phỏng). |
| `Duration` | Tổng thời gian chạy thực tế của toàn bộ quy trình (giây). |
| `Active` | Số tác vụ chạy đồng thời cao nhất (Peak Active), bảo đảm không vượt quá giới hạn `max_concurrency`. |
| `Cache Test` | Kiểm tra lần 2 trên bộ nhớ đệm SQLite: tỷ lệ trúng cache đạt 100%, số lần gọi AI bằng 0. |

---

## 9. Giấy phép và quyền riêng tư

Phần mềm được phát hành theo giấy phép MIT. Các mô hình và công cụ bên ngoài (FFmpeg, Piper, Faster-Whisper, RapidOCR) giữ giấy phép mã nguồn mở nguyên bản của từng dự án.

Phần mềm không thu thập thông tin người dùng, không có hệ thống tài khoản máy chủ riêng và không gửi dữ liệu ra bên ngoài trừ khi người dùng chủ động cấu hình dịch vụ dịch trực tuyến.
