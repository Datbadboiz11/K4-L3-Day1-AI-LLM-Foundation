# K4 — Ngày 1: Bài Tập & Phản Ánh
## Khám Phá LLM API | Phiếu Thực Hành

**Thời lượng:** 4 tiếng
**Cách làm:** Trả lời từng câu ngay sau khi hoàn thành block tương ứng —
đừng để dồn hết về cuối buổi. Thay dòng `*Câu trả lời của bạn*` bằng câu
trả lời thật (chấm tự động sẽ đếm số câu đã trả lời).

---

## Block 1 — API Cơ Bản (trả lời sau Checkpoint 1)

### Câu 1.1 — Độ nhạy của temperature
Gọi `call_openai` với temperature 0.0, 0.5, 1.0 và 1.5 dùng prompt
**"Hãy kể cho tôi một sự thật thú vị về Việt Nam."**

**Bạn nhận thấy quy luật gì qua bốn phản hồi?** (2–3 câu)
> Khi temperature thấp như 0.0, câu trả lời ổn định và khá an toàn, tập trung vào một sự thật quen thuộc là hang Sơn Đoòng. Khi tăng lên 0.5 và 1.0, nội dung vẫn cùng chủ đề nhưng cách diễn đạt, chi tiết và độ phong phú thay đổi nhẹ. Ở temperature 1.5, model bắt đầu chọn một hướng khác hẳn là hạt điều, cho thấy temperature càng cao thì phản hồi càng đa dạng và khó đoán hơn.

### Câu 1.2 — Chọn temperature cho sản phẩm
**Bạn sẽ đặt temperature bao nhiêu cho chatbot hỗ trợ khách hàng, và tại sao?**
> Với chatbot hỗ trợ khách hàng, em sẽ đặt temperature khoảng 0.2 hoặc 0.3 để câu trả lời ổn định, nhất quán và ít bịa hơn. Loại chatbot này cần ưu tiên độ chính xác, đúng chính sách và dễ kiểm soát hơn là sáng tạo. Nếu cần giọng văn tự nhiên hơn một chút thì có thể tăng nhẹ, nhưng không nên đặt quá cao.

### Câu 1.3 — Đánh đổi chi phí
Kịch bản: 10.000 người dùng hoạt động mỗi ngày, mỗi người gọi API 3 lần,
mỗi lần trung bình ~350 token đầu ra.

**Ước tính GPT-4o đắt hơn GPT-4o-mini bao nhiêu lần cho workload này? Nêu một
trường hợp GPT-4o xứng đáng với chi phí và một trường hợp nên dùng mini:**
> Workload có 10.000 × 3 × 350 = 10.500.000 token đầu ra mỗi ngày. Với giá output trong bài, GPT-4o tốn khoảng 105 USD/ngày, còn GPT-4o-mini khoảng 6,3 USD/ngày, tức GPT-4o đắt hơn khoảng 16,7 lần. GPT-4o xứng đáng dùng cho các tác vụ cần lập luận khó, phân tích sâu hoặc câu trả lời chất lượng cao; còn GPT-4o-mini phù hợp cho chatbot FAQ, tóm tắt ngắn hoặc các tác vụ đơn giản cần tối ưu chi phí.

---

## Block 2 — System Prompt & Token (trả lời sau Checkpoint 2)

### Câu 2.1 — Sức mạnh của persona
Gọi `chat_with_system_prompt` hai lần với cùng câu hỏi
**"Giải thích blockchain là gì?"** nhưng hai system prompt khác nhau:
- "Bạn là giáo viên tiểu học, giải thích thật đơn giản cho trẻ 8 tuổi."
- "Bạn là chuyên gia tài chính, trả lời chuyên sâu bằng thuật ngữ kỹ thuật."

**Hai phản hồi khác nhau như thế nào (độ dài, từ vựng, ví dụ)? System prompt
ảnh hưởng đến hành vi model ra sao?** (3–4 câu)
> Phản hồi với vai trò giáo viên tiểu học dùng ngôn ngữ đơn giản, gần gũi và lấy ví dụ “cuốn sổ ghi chép của cả lớp” để trẻ 8 tuổi dễ hình dung. Phản hồi với vai trò chuyên gia tài chính dài hơn, có cấu trúc rõ hơn và dùng nhiều thuật ngữ kỹ thuật như “cấu trúc dữ liệu phân tán”, “phi tập trung”, “sổ cái kỹ thuật số”, “hash” và “nút trong mạng”. Điều này cho thấy system prompt ảnh hưởng trực tiếp đến cách model chọn giọng văn, mức độ chi tiết, từ vựng và kiểu ví dụ. Cùng một câu hỏi, nhưng persona khác nhau làm model điều chỉnh câu trả lời cho đúng đối tượng người đọc.

### Câu 2.2 — tiktoken vs đếm từ
Chọn một đoạn văn tiếng Việt ~100 từ. So sánh số token theo `count_tokens`
(tiktoken) với ước lượng `số từ / 0.75` mà Part 1 đã dùng.

**Hai con số chênh nhau bao nhiêu phần trăm? Vì sao tiếng Việt thường tốn
nhiều token hơn tiếng Anh cùng độ dài?**
> Em chọn một đoạn văn tiếng Việt dài 91 từ. Hàm count_tokens đếm được 101 token, còn cách ước lượng số từ / 0.75 cho ra khoảng 121,33 token, chênh lệch khoảng 16,76%. Hai con số khác nhau vì token không trùng hoàn toàn với từ: dấu tiếng Việt, dấu câu, khoảng trắng và cách tokenizer tách từng cụm ký tự đều ảnh hưởng đến số token. Tiếng Việt thường có nhiều dấu và có thể bị tách thành nhiều mảnh token hơn tiếng Anh, nên dùng tokenizer như tiktoken sẽ đáng tin cậy hơn so với ước lượng bằng số từ.

---

## Block 3 — Streaming & Độ Bền (trả lời sau Checkpoint 3)

### Câu 3.1 — Trải nghiệm người dùng với streaming
**Streaming quan trọng nhất trong trường hợp nào, và khi nào thì
non-streaming lại phù hợp hơn?** (1 đoạn văn)
> Streaming quan trọng nhất khi câu trả lời dài hoặc người dùng cần cảm giác hệ thống đang phản hồi ngay, ví dụ chatbot tư vấn, trợ lý học tập, tạo nội dung dài hoặc giải thích từng bước. Khi stream, người dùng có thể đọc phần đầu trước thay vì chờ toàn bộ câu trả lời hoàn tất, nên trải nghiệm tự nhiên hơn. Non-streaming phù hợp hơn với các tác vụ cần nhận kết quả hoàn chỉnh một lần để xử lý tiếp, ví dụ phân loại văn bản, trích xuất JSON, chấm điểm tự động hoặc gọi API nội bộ mà người dùng không trực tiếp nhìn thấy quá trình sinh câu trả lời.

### Câu 3.2 — Vì sao backoff theo cấp số nhân?
**So với delay cố định (ví dụ luôn chờ 1 giây), exponential backoff có lợi
thế gì khi API bị quá tải? Điều gì xảy ra nếu hàng nghìn client cùng retry
với delay cố định giống nhau?**
> Exponential backoff tốt hơn delay cố định vì khi API đang quá tải, mỗi lần retry sau sẽ chờ lâu hơn, giúp giảm áp lực lên server và tăng khả năng request thành công. Nếu tất cả client đều retry với cùng một delay cố định, ví dụ luôn 1 giây, hàng nghìn request có thể cùng quay lại server đúng một thời điểm và tạo thêm một đợt quá tải mới. Backoff theo cấp số nhân giúp các lần thử lại được giãn ra dần, làm hệ thống ổn định hơn khi gặp lỗi tạm thời hoặc nghẽn tải.

---

## Block 4 — Mini-Project (trả lời sau Checkpoint 4)

### Câu 4.1 — Thiết kế persona
**Bạn chọn persona gì cho trợ lý của mình? Viết lại system prompt đó và giải
thích 1–2 lựa chọn từ ngữ quan trọng trong prompt (ví dụ: vì sao yêu cầu
"trả lời ngắn gọn", vì sao chỉ định ngôn ngữ...):**
> Em chọn persona: “Bạn là trợ giảng AI thân thiện của khóa học AI, trả lời ngắn gọn, dễ hiểu bằng tiếng Việt và ưu tiên ví dụ thực tế.” Em dùng cụm “trợ giảng AI thân thiện” để model giữ giọng hỗ trợ, gần gũi với người học thay vì trả lời quá khô cứng. Em yêu cầu “trả lời ngắn gọn, dễ hiểu bằng tiếng Việt” để câu trả lời phù hợp với sinh viên mới học LLM API và tránh lan man.

### Câu 4.2 — Hạn chế & cải thiện
**Trợ lý của bạn hiện có hạn chế lớn nhất là gì (ví dụ: history chỉ 3 lượt,
không có bộ nhớ dài hạn, không kiểm duyệt nội dung...)? Đề xuất một cải
thiện cụ thể và mô tả ngắn cách triển khai:**
> Hạn chế lớn nhất của trợ lý hiện tại là history chỉ giữ 3 lượt gần nhất, nên nếu cuộc trò chuyện dài thì model có thể quên thông tin quan trọng ở đầu phiên. Một cải thiện cụ thể là thêm cơ chế tóm tắt lịch sử: khi history vượt quá giới hạn, chương trình dùng model hoặc một hàm riêng để tóm tắt các lượt cũ thành một system/context message ngắn. Cách này giúp giữ được ý chính dài hạn mà vẫn không làm input token tăng quá nhiều.

---

## Danh Sách Kiểm Tra Nộp Bài

- [ ] `python grade.py` — xem điểm tự động, mục tiêu ≥ 75/100
- [ ] Cả 4 checkpoint pytest đều pass
- [ ] Tất cả 9 câu trong file này đã được trả lời
- [ ] Đã copy bài làm vào folder `solution/`, push lên fork và dán link trên trang bài Lab ở VLearn trước 23:59 ngày 11/09/2026
