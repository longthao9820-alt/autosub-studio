# AutoSub Studio

Phần mềm Windows để **tách phụ đề, dịch phụ đề, lồng tiếng và render video** — chạy trên
máy của bạn, không gửi dữ liệu lên máy chủ riêng của ai.

---

## 1. Mở phần mềm

**Cách nhanh nhất:** vào thư mục `dist\AutoSubStudio\` rồi **nhấp đúp vào `AutoSubStudio.exe`**.

Lần đầu mở, Windows có thể hiện bảng "Windows protected your PC" vì phần mềm chưa có chữ ký
số. Bấm **More info** → **Run anyway**.

Muốn mở nhanh hơn về sau: nhấp chuột phải vào `AutoSubStudio.exe` → **Send to** →
**Desktop (create shortcut)**.

**Không cần cài gì thêm.** Bản đóng gói đã chứa sẵn FFmpeg, mô hình nhận dạng giọng nói
(`faster-whisper-small`, 464 MB), mô hình đọc chữ trên hình, và thư viện CUDA để chạy
bằng card đồ họa.

---

## 1a. Tăng tốc bằng card đồ họa

Phần mềm **tự phát hiện và dùng card đồ họa** của máy, không cần bật gì. Đo trên máy có
card NVIDIA RTX 3060:

| Việc | Bằng CPU | Bằng card đồ họa | Nhanh hơn |
|---|---|---|---|
| Nhận dạng giọng nói (clip 99 giây) | 21,5 giây | 6,0 giây | **3,6 lần** |
| Đọc chữ trên hình — OCR (30 khung hình) | 42,5 giây | 9,0 giây | **4,7 lần** |
| Render phụ đề cứng (video 1080p 30 giây) | 12,0 giây | 3,0 giây | **4,0 lần** |

Máy nào dùng được gì:

| Loại card | Nhận dạng giọng nói | Đọc chữ (OCR) | Render video |
|---|---|---|---|
| NVIDIA (có driver) | ✅ nhanh hơn ~3,6 lần | ✅ nhanh hơn ~4,7 lần | ✅ NVENC |
| Intel (chip đồ họa tích hợp) | ❌ chạy bằng CPU | ❌ chạy bằng CPU | ✅ Quick Sync |
| AMD | ❌ chạy bằng CPU | ❌ chạy bằng CPU | ✅ AMF |
| Không có card rời | ❌ chạy bằng CPU | ❌ chạy bằng CPU | ❌ chạy bằng CPU |

Trước khi giao việc cho card đồ họa, phần mềm **chạy thử một tấm ảnh nhỏ**. Máy nào thiếu
thư viện hoặc card trở chứng thì nó tự đọc bằng CPU, không báo lỗi giữa chừng.

**Điều kiện duy nhất:** máy đích phải có sẵn **driver card đồ họa** (máy dùng bình thường
thì đã có). Thư viện CUDA đã nằm trong gói nên không phải cài CUDA Toolkit.

Nếu máy không có card phù hợp, phần mềm **tự chuyển sang CPU**, chỉ chậm hơn chứ không
báo lỗi. Muốn xem máy đang dùng gì: tab **Cài đặt chung** có dòng trạng thái, hoặc nhấp
đúp `KIEM-TRA-MAY-NAY.bat`.

Muốn tắt tăng tốc (ví dụ khi cần tiết kiệm điện hoặc card đang bận việc khác): tab
**Cài đặt chung**, bỏ chọn hai ô *Nhận dạng giọng nói bằng card đồ họa* và *Render video
bằng phần cứng*.

> Lưu ý: render bằng card đồ họa cho tệp **nặng hơn** một chút so với CPU ở cùng mức chất
> lượng. Muốn tệp nhỏ lại thì tăng *CRF* ở tab B4 lên 24–26.

---

## 1b. Mang sang máy khác

Chép **nguyên cả thư mục** `dist\AutoSubStudio` sang máy Windows kia (USB, ổ cứng ngoài
hoặc qua mạng), rồi nhấp đúp `AutoSubStudio.exe` là chạy ngay — không cài đặt, không tải
thêm gì.

Dự án và cài đặt của bạn nằm trong thư mục `Data` bên trong, nên đi theo cùng.

Muốn chắc chắn máy mới chạy được, có hai tệp kiểm tra sẵn trong thư mục:

| Tệp | Làm gì | Mất bao lâu |
|---|---|---|
| `KIEM-TRA-MAY-NAY.bat` | Kiểm tra nhanh: có đủ FFmpeg, mô hình, thư mục ghi được không | ~10 giây |
| `KIEM-TRA-SAU.bat` | Chạy thử thật: đọc chữ trên ảnh, nhận dạng một câu nói, render một video ngắn | 1–2 phút |

Kết quả hiện lên màn hình và lưu vào `ket-qua-kiem-tra.txt` cạnh tệp chạy. Mỗi dòng có
nhãn `DAT` (được), `THIEU` (thiếu tính năng phụ) hoặc `HONG` (phải sửa mới chạy được).

Nếu chép phần mềm vào chỗ không ghi được (ví dụ ổ đĩa chỉ đọc), phần mềm tự chuyển sang
lưu ở `Documents` và `AppData` như bình thường.

---

## 2. Làm một video từ đầu đến cuối

Có sẵn một dự án mẫu tên **"Video mau de thu"** để bạn bấm thử ngay.

1. **Tạo dự án** — tab *Danh sách dự án* → bấm **Tạo dự án mới** → chọn tệp video.
2. **Lấy phụ đề** — tab *B1: Tách sub* → bấm **Lấy phụ đề bằng giọng nói**.
   - Chạy ngay, không cần mạng (mô hình đã có sẵn trong gói).
   - Model càng lớn càng chính xác nhưng càng chậm. `small` là mức cân bằng tốt.

   **Nếu video đã có chữ cháy sẵn trên hình (hardsub)**, dùng OCR thay vì giọng nói:

   1. Mở dự án — video hiện hình ngay và **khung xanh có sẵn** ở phần dưới màn hình
      *Screen Edit*, đúng chỗ phụ đề hay nằm. Không phải bật tắt gì cả.
   2. **Kéo giữa khung** để di chuyển đến chỗ có chữ. **Kéo 8 nút xanh** ở viền để
      phóng to hoặc thu nhỏ. Trong lúc kéo, phần ngoài khung mờ đi cho dễ ngắm; buông
      tay là sáng lại. Kích thước hiện ngay trên khung.
   3. Bấm **Lấy phụ đề bằng OCR**.

   OCR mặc định dùng **PP-OCRv6 Medium** chuyên đọc tiếng Trung ở chế độ
   **Chính Xác**. Chế độ này lấy 6 khung hình/giây, bỏ kết quả dưới 50%, đọc
   lại toàn bộ từng dòng chữ và đối chiếu 3 khung gần nhau trước khi chốt câu.
   Nếu cần chạy nhanh hơn, chọn **Cân Bằng** hoặc **Nhanh**; các lựa chọn này
   thay đổi mô hình và tham số xử lý thật, không chỉ thay tên trên giao diện.

   Khung xanh chỉ có ở **Screen Edit**. Tab **Screen Render** để xem lại thành phẩm nên
   không có khung nào che tầm nhìn.

   Vùng đã khoanh được lưu theo dự án, lần sau mở lại vẫn còn. Nếu bạn đổi sang video
   có kích thước khác, phần mềm tự co vùng cho vừa khung hình mới. Muốn về vị trí ban
   đầu thì bấm **Đặt Lại Vùng Mặc Định**.

   **Cấu hình mặc định giống NTS.** Phần mềm đưa ảnh gốc vào PP-OCRv4, không tự đo hay
   ép lọc theo màu và không tự bỏ chữ đứng yên. Vì vậy video thông thường chỉ cần khoanh
   sát dòng phụ đề rồi chạy, không cần chỉnh thêm.

   - **Lọc theo màu là tùy chọn chỉ định.** Ô màu để trống nghĩa là không lọc. Chỉ khi
     video có logo/chữ quảng cáo khác màu nằm trong vùng OCR, bấm **Chọn Màu...** hoặc
     nhập mã `#RRGGBB`; lúc đó bộ lọc mới hoạt động. Bấm **Clear** để tắt và quay lại
     ảnh gốc. Video nét để sai lệch 10-15%; video mờ có thể tăng lên.
   - **Theo chiều cao chữ.** Ô **Loại Bỏ Chiều Cao Chữ: Nhỏ Hơn < ... Và Lớn Hơn > ...**
     bỏ những dòng quá thấp (chữ ghi chú nhỏ) hoặc quá cao (chữ tiêu đề to). Mặc định
     là 25-200 px giống dự án NTS của bạn. Nên khoanh vùng sát phụ đề trước khi thay số;
     bấm **Đo Chiều Cao Chữ** nếu muốn phần mềm đo từ video.

   Hai thanh **Chỉnh Sáng** và **Chỉnh Tương Phản** dùng khi video quá tối hoặc quá bạc
   làm chữ chìm vào nền; kéo lên một chút là chữ rõ hơn trước khi đọc.

   Đo trên một video dựng sẵn (nền nhiều màu, nén mạnh, có logo trắng cùng màu phụ đề và
   một dòng chữ quảng cáo vàng nằm ngay trong vùng khoanh): trước khi lọc, phụ đề đọc ra
   **giống bản gốc 77%** và **12 câu** trong đó 6 câu là rác của logo; sau khi lọc còn
   **đúng 6 câu**, **giống bản gốc 99,5%**.

   **Mốc thời gian đo hai lần cho khớp.** Chế độ Chính Xác quét 6 khung/giây, sau đó
   lấy thêm các khung sát nhau quanh đầu và
   cuối mỗi câu để tìm đúng lúc chữ hiện ra và biến mất. Đo trên video có mốc biết trước:
   sai số trung bình giảm từ **257 mili giây xuống 50 mili giây**, đổi lại thời gian chạy
   dài hơn khoảng gấp đôi. Muốn nhanh hơn và chấp nhận mốc thô: bỏ chọn **Đo lại mốc thời
   gian cho chính xác** ở tab *B1*.
   **Tách sub xong là có ngay tệp `.srt`.** Phần mềm tự lưu tệp phụ đề vào **đúng thư
   mục chứa video bạn đã thêm**, trùng tên với video (`tap 01.mp4` → `tap 01.srt`). Không
   phải bấm xuất tay. Những lần tách lại sau, phần mềm ghi đè đúng tệp đó chứ không rải
   thêm tệp mới. Nếu cạnh video đã sẵn có tệp `.srt` trùng tên từ trước, phần mềm để yên
   tệp đó và lưu thành `tap 01_1.srt` để không đè mất phụ đề của bạn. Trường hợp thư mục
   video không ghi được (ổ USB đã rút, thư mục chỉ đọc), tệp được lưu vào thư mục
   `exports` của dự án và ghi rõ trong Nhật ký.

   Muốn tắt: tab *Cài đặt chung* → bỏ chọn **Tách sub xong thì lưu luôn tệp .srt cạnh
   video**. Lúc đó vẫn xuất tay được bằng nút xuất phụ đề ở tab *B4*.
3. **Sửa lại phụ đề** — bảng phụ đề ở góc trên bên phải. Nhấp một dòng để video nhảy tới
   đúng chỗ đó. Bấm **Kiểm tra time** để phần mềm dò các câu chồng lấn, quá nhanh,
   hoặc bị trống.
4. **Dịch** — tab *B2: Dịch nội dung* → chọn ngôn ngữ đích → bấm **Dịch toàn bộ dự án**.
5. **Lồng tiếng** (nếu cần) — mở **VoiceStudio**, sau đó vào tab *B3: Ghép Giọng Đọc*
   → chọn **VoiceStudio Local** → **Kết Nối VoiceStudio / Nạp Giọng**. Tool chỉ hiển thị
   giọng English/American: 8 giọng Kitten English nam/nữ, các profile English trong
   VoiceStudio và `Edge English US` làm dự phòng. Chọn giọng rồi bấm **START: Lồng
   Tiếng**. Chế độ mặc định tự tạo từng
   câu, canh vào time subtitle, hạ tiếng gốc xuống 20% khi giọng đọc xuất hiện và xuất
   luôn video `*_long_tieng.mp4` trong thư mục `exports` của dự án. Có thể chuyển sang
   **Chỉ Xuất Tệp Tiếng**, cho phép chồng tiếng, đặt khoảng nghỉ cuối, chỉnh tốc độ/cao
   độ/EQ, lưu từng voice hoặc thêm profile giọng **Mặc định/Nam/Nữ**.
6. **Render** — tab *B4: Render và xuất* → **Render video**. Tệp kết quả nằm trong
   thư mục `exports` của dự án.

### Nhấp chuột phải vào dòng dự án

Bảng dự án có menu chuột phải làm được gần hết mọi việc mà không cần đi vòng qua các tab:

- **Mở dự án**
- **Tách sub bằng chữ trên hình (OCR)** / **Tách sub bằng giọng nói**
- **Dịch phụ đề**, **Phân tách giọng nam / nữ**, **Lồng tiếng theo bản dịch**
- **Che mờ phụ đề gốc**, **Render video**
- **Chạy kịch bản đang tích chọn**, **Xuất phụ đề .srt...**
- **Mở thư mục dự án**, **Xem nhật ký của dự án**, **Dừng các tác vụ đang chạy**
- **Xóa dự án...**

Chọn việc gì thì phần mềm tự mở dự án ở dòng đó trước rồi mới chạy, nên không cần nhấp
đúp mở dự án thủ công. Nhấp chuột phải vào chỗ trống trong bảng thì hiện **Tạo dự án mới**.

### Chạy hàng loạt

Cột **Các bước chạy tự động** ở góc dưới bên trái cho phép tích chọn nhiều bước rồi bấm
**Chạy kịch bản** để phần mềm làm liền một mạch. Bạn có thể đặt tên và lưu lại nhiều
kịch bản khác nhau.

---

## 3. Phím tắt

| Phím | Tác dụng |
|---|---|
| `Space` | Phát / tạm dừng video |
| `Ctrl+S` | Lưu dự án |
| `Ctrl+Z` / `Ctrl+Y` | Hoàn tác / làm lại |
| `Ctrl+K` | Tách câu tại vị trí video đang phát |
| `Ctrl+M` | Gộp các câu đang chọn |
| `Ctrl+N` | Thêm câu mới |
| `Ctrl+Delete` | Xóa các câu đang chọn |

---

## 4. Dữ liệu lưu ở đâu

Bản đóng gói chạy ở **chế độ di động**: mọi thứ nằm trong thư mục `Data` ngay cạnh
`AutoSubStudio.exe`.

| Nội dung | Vị trí (bản đóng gói) |
|---|---|
| Dự án, video, âm thanh, phụ đề, kết quả render | `Data\workspace\projects\` |
| Cơ sở dữ liệu danh sách dự án | `Data\workspace\db\app.db` |
| Cấu hình phần mềm | `Data\config\config.json` |
| Khóa API (mã hóa theo tài khoản Windows) | `Data\config\secrets.dat` |
| Nhật ký khi phần mềm gặp lỗi nặng | `Data\config\logs\crash.log` |
| FFmpeg, mô hình AI và thư viện CUDA đi kèm | `_internal\ffmpeg\`, `_internal\models\`, `_internal\cuda\` |

Khi **chạy từ mã nguồn** (không phải bản đóng gói), dữ liệu lưu ở
`C:\Users\<tên bạn>\Documents\AutoSubStudio\` và cấu hình ở
`C:\Users\<tên bạn>\AppData\Roaming\AutoSubStudio\`.

Đổi thư mục lưu: tab **Cài đặt chung** → **Chọn thư mục làm việc...**

Mỗi dự án là một thư mục riêng gồm `video/`, `audio/`, `subtitles/`, `temp/`, `exports/`
và tệp `project.json` chứa toàn bộ phụ đề. Cứ 60 giây phần mềm tự ghi một bản phục hồi;
nếu bị tắt đột ngột, lần mở sau sẽ hỏi bạn có muốn khôi phục không.

---

## 5. Có sẵn gì, thiếu gì

**Đã có sẵn trong bản đóng gói, chạy được ngay không cần mạng:**
nhận dạng giọng nói, đọc chữ cháy sẵn trên hình (OCR), tách nhạc nền mức cơ bản,
lồng tiếng bằng giọng Windows, render video.

**Cần Internet khi dùng:** dịch bằng Google, dịch bằng Claude, giọng đọc Edge TTS.

**Hai tính năng chưa gói kèm** (vì rất nặng), muốn có thì cài thêm rồi đóng gói lại:

| Tính năng | Cách bật |
|---|---|
| Tách nhạc nền chất lượng cao (Demucs) | `pip install demucs` — khoảng 2 GB, nên có card đồ họa |
| Dịch bằng Claude | `pip install anthropic`, rồi nhập khóa API ở tab *Cài đặt chung* |

Chạy trong PowerShell tại thư mục dự án:

```powershell
.\.venv\Scripts\python.exe -m pip install demucs
powershell -ExecutionPolicy Bypass -File scripts\build.ps1
```

Không cài thì các nút tương ứng vẫn hiện nhưng mờ đi kèm dòng chữ giải thích, chứ không
báo lỗi.

### Đóng gói lại

Lệnh `scripts\build.ps1` lấy các thành phần nhúng từ thư mục `assets`:

```
assets\ffmpeg\ffmpeg.exe          (bắt buộc)
assets\ffmpeg\ffprobe.exe         (bắt buộc)
assets\models\faster-whisper-small\model.bin   (mô hình nhận dạng)
assets\cuda\*.dll                 (tăng tốc GPU — bỏ đi thì chạy bằng CPU)
```

Bộ thư viện CUDA lấy từ các gói `nvidia-*-cu12`, đã lọc bỏ những tệp không cần
(2,3 GB → 2,0 GB):

```powershell
.\.venv\Scripts\python.exe -m pip install nvidia-cublas-cu12 nvidia-cudnn-cu12 `
    nvidia-cuda-runtime-cu12 nvidia-cufft-cu12 nvidia-cuda-nvrtc-cu12
```

rồi chép 16 tệp sau từ `.venv\Lib\site-packages\nvidia\*\bin\` vào `assets\cuda\`:

- Cho nhận dạng giọng nói: `cublas64_12`, `cublasLt64_12`, `cudnn64_9`,
  `cudnn_graph64_9`, `cudnn_ops64_9`, `cudnn_cnn64_9`, `cudnn_heuristic64_9`,
  `cudnn_engines_tensor_ir64_9`, `cudnn_ext64_9`, `cudnn_engines_runtime_compiled64_9`,
  `nvrtc64_120_0`, `nvrtc-builtins64_129`.
- Thêm cho OCR: `cudart64_12`, `cufft64_11`, `nvJitLink_120_0`, và
  `cudnn_engines_precompiled64_9` (522 MB — thiếu tệp này thì OCR trên GPU báo lỗi
  `CUDNN_STATUS_INTERNAL_ERROR`, dù nhận dạng giọng nói vẫn chạy bình thường).

Phần OCR trên GPU dùng `onnxruntime-gpu==1.22.0`. **Đừng nâng lên 1.28** — bản đó đòi
CUDA 13 trong khi nhận dạng giọng nói (CTranslate2) cần CUDA 12, hai bên không dùng
chung được.

Muốn đổi sang mô hình lớn hơn cho chính xác hơn (nặng hơn, chậm hơn):

```powershell
.\.venv\Scripts\python.exe -c "from huggingface_hub import snapshot_download; snapshot_download('Systran/faster-whisper-medium', local_dir='assets/models/faster-whisper-medium')"
```

Rồi vào tab *Cài đặt chung* của phần mềm chọn model tương ứng, và đóng gói lại. Nếu thiếu
tệp nào trong `assets`, script sẽ cảnh báo rõ trước khi đóng gói.

---

## 6. Lỗi thường gặp

**"Chưa có FFmpeg"** — bản đóng gói đã kèm sẵn, nên lỗi này chỉ xảy ra khi chép thiếu tệp
lúc mang thư mục sang máy khác. Chạy `KIEM-TRA-MAY-NAY.bat` để xem thiếu gì; thư mục
`_internal\ffmpeg` phải có đủ `ffmpeg.exe` và `ffprobe.exe`.

**"Không tải được model vì không có mạng"** — chỉ xảy ra nếu bạn chọn model khác với
model đi kèm (`small`). Quay lại chọn `small` ở tab B1, hoặc nối mạng một lần để tải
model mới về.

**"Card đồ họa hết bộ nhớ"** — vào *Cài đặt chung*, bỏ chọn *Ưu tiên dùng card đồ họa*,
hoặc chọn model nhỏ hơn ở tab B1.

**Nhận dạng ra chữ sai nhiều** — thử model lớn hơn (`medium`, `large-v3`), hoặc chạy bước
*Xóa nhạc nền, giữ lời thoại* trước khi nhận dạng.

**Dịch báo lỗi mạng** — Google Dịch cần Internet. Có thể chọn *Không dịch (giữ nguyên)*
rồi tự gõ bản dịch vào cột bên phải.

**Câu lồng tiếng bị đọc quá nhanh** — nới mốc kết thúc của câu đó ra, hoặc rút gọn câu dịch
cho ngắn lại. Cột *Tỉ lệ* trong bảng phụ đề cho biết bản dịch dài gấp mấy lần bản gốc.

**Bấm lồng tiếng nhưng thiếu một vài câu** — Giọng Free tự thử lại tối đa 3 lần khi server
không trả audio. Nếu vẫn lỗi, Nhật ký và thông báo hoàn tất sẽ ghi đúng số câu thất bại;
thử lại khi mạng ổn định. Tool không còn âm thầm coi câu lỗi là đã hoàn thành.

**VoiceStudio Local chưa sẵn sàng** — bấm **Kết Nối VoiceStudio / Nạp Giọng**. Tool tự
mở bản VoiceStudio đã cài tại `%LOCALAPPDATA%\Programs\VoiceStudio` và kết nối API local
`127.0.0.1:3900`; không gửi nội dung ra dịch vụ đám mây. Nếu không muốn dùng VoiceStudio,
chọn `Edge English US`.

**Lưu nhiều bộ cấu hình như NTS** — tại tab *Cấu Hình Chung*, chỉnh các thông số mong
muốn, nhập tên rồi bấm **Tạo Mới**. Chọn tên trong **Cấu Hình Tùy Chỉnh** để đổi toàn bộ
thiết lập; bấm **Lưu** để cập nhật profile đang chọn. Nút **Xóa** chỉ xóa profile đang
chọn và không thể xóa `default`.

**Render rất lâu** — vào tab B4 đổi *Tốc độ mã hóa* sang `veryfast` và tăng *CRF* lên 24–26.

**OCR đọc thiếu chữ hoặc lẫn tạp** — khoanh vùng sát vào dòng chữ hơn, chừa lề vừa đủ.
Vùng càng gọn thì đọc càng chính xác và càng nhanh. Cách khoanh: xem mục 2 ở trên.

**OCR vẫn bắt lẫn logo, chữ quảng cáo** — tab *B1*, bấm **Chọn Màu...** để chỉ định đúng
màu phụ đề. Nếu logo cùng màu nhưng nhỏ hơn, đặt ô **Nhỏ Hơn <** gần cỡ chữ phụ đề (ví dụ
phụ đề cao 44 px, logo cao 30 px thì đặt 36).

**OCR đọc mất chữ sau khi lọc màu** — bấm **Clear** để tắt lọc và dùng lại ảnh gốc như
mặc định NTS; hoặc chọn lại đúng màu rồi tăng sai lệch lên 20–25%.

**Phần mềm tự tắt** — mở tệp `crash.log` (đường dẫn ở mục 4) để xem chi tiết. Công việc
đang làm thường vẫn còn trong bản tự lưu phục hồi.

**Không rõ máy có chạy được không** — nhấp đúp `KIEM-TRA-SAU.bat`. Nó chạy thử thật ba
việc: đọc chữ trên ảnh, nhận dạng một câu nói, và render một video ngắn.

---

## 7. Dành cho người muốn sửa mã nguồn

```powershell
cd <thư mục dự án>
.\.venv\Scripts\python.exe -m autosub_studio          # chạy từ mã nguồn
.\.venv\Scripts\python.exe -m pytest -q               # chạy kiểm thử
.\.venv\Scripts\python.exe -m ruff check .            # kiểm tra chất lượng mã
.\.venv\Scripts\python.exe -m mypy                    # kiểm tra kiểu dữ liệu
.\.venv\Scripts\python.exe scripts\smoke_media.py     # thử toàn bộ luồng media
.\.venv\Scripts\python.exe scripts\smoke_speech.py    # thử giọng đọc + nhận dạng
.\.venv\Scripts\python.exe -m autosub_studio --selftest-full   # tự kiểm tra sâu
powershell -ExecutionPolicy Bypass -File scripts\build.ps1     # đóng gói lại
```

Cấu trúc mã nguồn:

```
src/autosub_studio/
├─ core/        Kiểu dữ liệu, đọc/ghi SRT-VTT-ASS, thao tác biên tập
├─ data/        Cơ sở dữ liệu SQLite và tệp project.json
├─ services/    FFmpeg, xử lý media, hàng đợi tác vụ, cấu hình
├─ providers/   Nhận dạng giọng nói, OCR (kèm bộ lọc màu chữ `ocr_filter`),
│               dịch, giọng đọc, tách nhạc
├─ pipeline/    11 bước nghiệp vụ dùng chung cho nút bấm và kịch bản tự động
└─ ui/          Giao diện PySide6
```

---

## 8. Giấy phép và quyền riêng tư

Mã nguồn viết mới hoàn toàn, phát hành theo giấy phép MIT. FFmpeg và các mô hình AI là
thành phần bên ngoài và giữ giấy phép riêng của chúng.

Các dự án mã nguồn mở chính được dùng trong luồng nhận dạng:

- [RapidOCR](https://github.com/RapidAI/RapidOCR) (Apache-2.0): OCR Trung Quốc chạy cục bộ.
- [RapidFuzz](https://github.com/rapidfuzz/RapidFuzz) (MIT): đối chiếu nhanh các biến thể
  chữ OCR giữa nhiều khung hình.
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (MIT): nhận dạng giọng nói,
  Silero VAD và timestamp theo từng từ.
- [charset-normalizer](https://github.com/jawah/charset_normalizer) (MIT): đọc các tệp phụ
  đề dùng bảng mã cũ như GB18030 và Big5.

Phần mềm không thu thập dữ liệu, không có tài khoản, không gửi thống kê. Chỉ ba việc sau
mới cần Internet, và đều do bạn chủ động bấm: tải mô hình nhận dạng lần đầu, dịch bằng
Google hoặc Claude, và giọng đọc Edge TTS.
