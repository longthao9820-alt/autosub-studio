# AutoSub Studio V2

Phần mềm Windows để **tách phụ đề, dịch phụ đề, lồng tiếng offline và render video** — chạy trực tiếp trên
máy của bạn, bảo đảm an toàn dữ liệu và quyền riêng tư.

---

## 1. Mở phần mềm

**Cách nhanh nhất:** vào thư mục `dist\AutoSubStudio\` rồi **nhấp đúp vào `AutoSubStudio.exe`**.

Lần đầu mở, Windows có thể hiện bảng "Windows protected your PC" vì phần mềm chưa có chữ ký
số. Bấm **More info** → **Run anyway**.

Muốn mở nhanh hơn về sau: nhấp chuột phải vào `AutoSubStudio.exe` → **Send to** →
**Desktop (create shortcut)**.

**Không cần cài đặt phức tạp.** Bản đóng gói chính thức đi kèm sẵn FFmpeg, mô hình nhận dạng giọng nói
(`faster-whisper-small`), mô hình đọc chữ trên hình (PP-OCR), runtime giọng đọc Piper Local (`_internal\piper\`),
và các thư viện tăng tốc GPU.

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
3. Chọn hoặc cấu hình mô hình:
   - `sub`: tối ưu cho tốc độ và tác vụ dịch thuật phụ đề chuẩn xác.
   - `prime`: mô hình suy luận nâng cao cho các ngữ cảnh phức tạp.
   - Mức độ suy luận (*Thinking level*): tắt (`off`), thấp (`low`), vừa (`medium`), cao (`high`).
4. Bấm **Kiểm tra kết nối** để kiểm tra trực tiếp endpoint và danh sách model khả dụng.

---

## 3. Luồng làm việc 4 bước (B1 → B4)

### Bước 1: Tách phụ đề (B1)

- **Tách bằng giọng nói (ASR):** Sử dụng Faster-Whisper chạy trực tiếp trên máy (CPU hoặc GPU). Model mặc định kèm theo là `small`, nhận dạng chuẩn xác kèm mốc thời gian từng từ.
- **Tách bằng chữ trên hình (OCR):**
  - Dành cho video có phụ đề cứng (hardsub).
  - Vùng nhận dạng trực quan tại màn hình *Screen Edit* với khung chữ nhật điều chỉnh linh hoạt.
  - Sử dụng engine PP-OCRv6 Medium hoặc PP-OCRv4 Mobile, hỗ trợ lọc màu chữ (`#RRGGBB`) và lọc chiều cao chữ để loại bỏ logo/quảng cáo.
  - Chế độ quét mốc thời gian 2 lần giúp mốc khớp chính xác với khung hình xuất hiện/biến mất của phụ đề.
  - Tự động xuất tệp phụ đề `.srt` cùng tên bên cạnh tệp video gốc.

### Bước 2: Dịch nội dung (B2)

Thao tác dịch xử lý trực tiếp trên danh sách câu phụ đề, không làm thay đổi timeline video:

- **Lựa chọn dịch:**
  - `Google (miễn phí)`: dịch nhanh qua Google Dịch (cần Internet).
  - `AI Gateway` / `Server AI API`: dịch thông minh qua AI Gateway với model `sub` hoặc `prime`, bảo toàn cấu trúc 1:1, hỗ trợ ngữ cảnh trước/sau.
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
| `Data\models\piper\` | Các mô hình giọng đọc offline đã tải |
| `Data\config\logs\crash.log` | Nhật ký sự cố khi xảy ra lỗi ngoài ý muốn |
| `_internal\` | FFmpeg, runtime Piper, mô hình ASR và thư viện GPU |

Khi chạy từ mã nguồn lập trình (không dùng bản đóng gói), dữ liệu lưu tại thư mục người dùng:
`Documents\AutoSubStudio\` và `AppData\Roaming\AutoSubStudio\`.

---

## 6. Yêu cầu kết nối mạng

- **Không cần Internet (chạy hoàn toàn offline):**
  - Nhận dạng giọng nói (Whisper).
  - Đọc chữ trên hình ảnh (OCR).
  - Tách nhạc nền cơ bản.
  - Lồng tiếng với Local Voice (Piper Local) khi mô hình đã có trên máy.
  - Render và xuất video phụ đề cứng.
- **Cần kết nối Internet:**
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
assets\piper\piper.exe
assets\piper\espeak-ng-data\
assets\models\faster-whisper-small\
assets\cuda\*.dll
```

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
```

---

## 9. Giấy phép và quyền riêng tư

Phần mềm được phát hành theo giấy phép MIT. Các mô hình và công cụ bên ngoài (FFmpeg, Piper, Faster-Whisper, RapidOCR) giữ giấy phép mã nguồn mở nguyên bản của từng dự án.

Phần mềm không thu thập thông tin người dùng, không có hệ thống tài khoản máy chủ riêng và không gửi dữ liệu ra bên ngoài trừ khi người dùng chủ động cấu hình dịch vụ dịch trực tuyến.
