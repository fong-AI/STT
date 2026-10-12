# Meeting STT local — thiết kế bản thử

## Mục tiêu
Người dùng tham gia Google Meet hoặc Zoom như bình thường, mở công cụ bên cạnh để đọc transcript trong cuộc họp. STT chạy local, không có phí API theo phút. Codex xây dựng công cụ. Dự án riêng tại C:/Users/PHONG/meeting-stt, không sửa MMVN hoặc credential Google.

## Lựa chọn
Chọn faster-whisper vì tích hợp Python đơn giản và hỗ trợ CPU INT8. whisper.cpp là lựa chọn thay thế nhưng cần tích hợp executable/binding. Cloud STT không đáp ứng mục tiêu không tính phí API.
Máy đã kiểm tra: Windows, Python 3.12, i5-12600K, RTX 3060. Bắt đầu bằng model đa ngôn ngữ small trên CPU INT8; chỉ cân nhắc CUDA sau khi đo hiệu năng. Model nhỏ hơn có thể giảm độ chính xác tiếng Việt.

## Luồng và giao diện
PyAudioWPatch thu WASAPI loopback của đầu ra Zoom/Meet và micro được chọn. Giữ hai nguồn riêng để gắn nhãn Bạn / Cuộc họp; chưa xác định từng người phía bên kia.
Chuyển audio từ sample rate thực của thiết bị về mono float32 16 kHz bằng numpy và scipy.signal.resample_poly. Gom từng đoạn tối đa khoảng 5 giây rồi đưa vào queue giới hạn. Một worker chạy model faster-whisper dùng chung, có VAD khi nhận dạng để giảm suy diễn trong im lặng. Capture và UI không bị inference chặn.
UI Tkinter có chọn đầu ra loopback, micro (có thể tắt), tiếng Việt/Anh/tự nhận ngôn ngữ, Start/Stop, mức audio, trạng thái và transcript với timestamp và nhãn nguồn. Bản thử cập nhật theo đoạn ngắn, không cam kết caption theo từng từ hoặc độ trễ cố định.

## Vòng đời và dữ liệu
Chỉ thu sau khi người dùng bấm Start. Lần đầu cần Internet tải dependency/model; sau đó model cache chạy offline. Không dùng Google/OpenAI credential, không gọi STT API.
Mỗi phiên tạo file transcript UTF-8 riêng trong transcripts/, append và flush từng kết quả; không lưu audio thô. Stop ngừng capture, gửi đoạn cuối tới worker và chờ xử lý xong. Không cho Start phiên khác trước khi phiên cũ hoàn tất. Đóng cửa sổ thực hiện cùng quy trình Stop.

## Lỗi và giới hạn
Nếu queue đầy, hiển thị lỗi và dừng capture, không bỏ audio âm thầm hoặc tăng RAM vô hạn. Lỗi thiết bị, tải model hoặc ghi file phải hiện rõ và giải phóng stream.
Loopback thu toàn bộ audio trên đầu ra được chọn, gồm cả ứng dụng khác. Dùng tai nghe để tránh audio cuộc họp lọt vào micro gây lặp chữ. Mute trong Zoom/Meet không tắt micro của công cụ: UI cần có nút tắt micro riêng. Đổi thiết bị giữa phiên cần Stop và chọn lại.
Chưa biết CPU xử lý hai nguồn có theo kịp hay không. Đo tốc độ trước khi hứa realtime. Whisper có thể nhận sai hoặc suy diễn chữ từ âm thanh nền.

## File và phạm vi thay đổi
- meeting_stt.py: UI, capture, worker STT và lifecycle.
- audio_utils.py: mono/resample và gom đoạn, kiểm tra độc lập.
- test_audio_utils.py: stereo, resample, thứ tự sample và đoạn dư cuối phiên.
- requirements.txt: faster-whisper, PyAudioWPatch, numpy, scipy với phiên bản đã thử.
- README.md, run.bat, .gitignore: hướng dẫn, khởi chạy và loại transcript/cache khỏi Git.
Không cần server, database, đăng nhập, bot meeting, nhận diện tên người nói, summary hoặc EXE installer.

## Kiểm chứng
Viết test đỏ trước logic audio rồi chạy test xanh. Kiểm tra cú pháp/import trong môi trường riêng. Kiểm tra liệt kê thiết bị và suy luận im lặng không tạo transcript. Thử loopback bằng audio có lời khi có nguồn thử phù hợp; không tự thu meeting/micro người dùng ngầm. Kiểm tra UI, Start/Stop, lỗi và file transcript. Đo thời gian inference so với thời lượng audio. Nếu chưa kiểm tra meeting thật, ghi rõ khi bàn giao.

## Trạng thái
Thiết kế đã tự rà soát phạm vi, lifecycle, bảo mật audio và xử lý lỗi; không có placeholder. Chờ người dùng duyệt thiết kế trước khi viết kế hoạch và triển khai.
