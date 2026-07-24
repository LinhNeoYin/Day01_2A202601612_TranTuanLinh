# K4 — Ngày 1: Bài Tập & Phản Ánh
## Khám Phá LLM API | Phiếu Thực Hành

**Thời lượng:** 14h00–18h00
**Cách làm:** Trả lời từng câu ngay sau khi hoàn thành block tương ứng —
đừng để dồn hết về cuối buổi. Thay dòng `*Câu trả lời của bạn*` bằng câu
trả lời thật (chấm tự động sẽ đếm số câu đã trả lời).

---

## Block 1 — API Cơ Bản (trả lời sau Checkpoint 1)

### Câu 1.1 — Độ nhạy của temperature
Gọi `call_openai` với temperature 0.0, 0.7, 1.2 và 1.8 dùng prompt
**"Hãy kể cho tôi một sự thật thú vị về Hà Nội."**

**Bạn nhận thấy quy luật gì qua bốn phản hồi? Ở mức nào phản hồi bắt đầu
kém mạch lạc?** (2–3 câu)
> *Khi tăng temperature, phản hồi của mô hình trở nên đa dạng và sáng tạo hơn, nhưng đồng thời tính nhất quán cũng giảm. Ở mức 0.0, câu trả lời ổn định và thiên về thông tin chính xác; khoảng 0.7–1.2 vẫn khá mạch lạc nhưng phong phú hơn. Đến khoảng 1.8, phản hồi thường bắt đầu kém mạch lạc hơn, có xu hướng lan man hoặc đưa ra các ý ít liên quan*

### Câu 1.2 — Chọn temperature cho sản phẩm
**Bạn sẽ đặt temperature bao nhiêu cho trợ lý soạn thảo hợp đồng pháp lý,
và bao nhiêu cho trợ lý viết slogan quảng cáo? Giải thích khác biệt.**
> *Nếu là trợ lý soạn thảo hợp đồng pháp lý thì mình sẽ để temperature khoảng 0.1 hoặc 0.2, vì mình muốn AI trả lời ổn định, chính xác và hạn chế tự sáng tạo để tránh làm thay đổi ý nghĩa của điều khoản. Còn nếu là trợ lý viết slogan quảng cáo thì mình sẽ để khoảng 1.2, vì lúc này cần nhiều ý tưởng mới và cách diễn đạt sáng tạo hơn để slogan hấp dẫn, độc đáo*

### Câu 1.3 — Đánh đổi chi phí
Kịch bản: 20.000 người dùng hoạt động mỗi ngày, mỗi người gọi API 2 lần,
mỗi lần trung bình ~500 token đầu ra.

**Ước tính chi phí mỗi ngày của model lớn so với model nhỏ cho workload này
(dựa trên bảng giá trong template). Nêu một trường hợp model lớn xứng đáng
với chi phí và một trường hợp model nhỏ là lựa chọn đúng:**
> *Với 20.000 người dùng, mỗi người gọi API 2 lần và mỗi lần sinh khoảng 500 token đầu ra, tổng lượng token mỗi ngày là khoảng 20 triệu token. Theo bảng giá trong template, chi phí của GPT-4o khoảng 300 USD/ngày, còn GPT-4o Mini khoảng 12 USD/ngày, tức model lớn đắt hơn khoảng 25 lần.Theo mình, model lớn xứng đáng với chi phí trong các tác vụ cần độ chính xác và suy luận cao như phân tích tài liệu pháp lý hoặc hỗ trợ lập trình phức tạp. Ngược lại, model nhỏ là lựa chọn phù hợp cho các tác vụ đơn giản như chatbot chăm sóc khách hàng, hỏi đáp thông thường hoặc tóm tắt văn bản ngắn vì chi phí thấp nhưng vẫn đáp ứng tốt nhu cầu.*

---

## Block 2 — System Prompt & Token (trả lời sau Checkpoint 2)

### Câu 2.1 — Sức mạnh của persona
Gọi `chat_with_system_prompt` hai lần với cùng câu hỏi
**"Giải thích máy học (machine learning) là gì?"** nhưng hai system prompt
khác nhau:
- "Bạn là một nhà thơ, trả lời mọi thứ bằng hình ảnh ví von, tránh thuật ngữ."
- "Bạn là kỹ sư phần mềm senior, trả lời chính xác, có ví dụ code khi phù hợp."

**Hai phản hồi khác nhau như thế nào (giọng văn, độ dài, mức kỹ thuật)?
Từ đó rút ra system prompt điều khiển được những khía cạnh nào của phản hồi?**
(3–4 câu)
> *Hai phản hồi khác nhau khá rõ. Với system prompt "nhà thơ", AI trả lời ngắn gọn, dùng nhiều hình ảnh ví von và ngôn ngữ gần gũi, hầu như không sử dụng thuật ngữ kỹ thuật. Còn với system prompt "kỹ sư phần mềm senior", câu trả lời chi tiết hơn, sử dụng các thuật ngữ chuyên môn, giải thích chính xác và có thể kèm ví dụ code minh họa. Qua đó có thể thấy system prompt có thể điều khiển vai trò của AI, giọng văn, mức độ chi tiết, mức độ kỹ thuật và cách trình bày của phản hồi.*

### Câu 2.2 — tiktoken vs đếm từ
Chọn một đoạn văn tiếng Việt ~150 từ. So sánh số token theo `count_tokens`
(tiktoken) với ước lượng `số từ / 0.75` mà Part 1 đã dùng.

**Hai con số chênh nhau bao nhiêu phần trăm? Nếu dùng ước lượng thô để dự
toán ngân sách API cho ứng dụng tiếng Việt, bạn sẽ dự toán thiếu hay thừa —
và vì sao?**
> *Sau khi so sánh, số token của tiktoken cao hơn cách ước lượng khoảng 5%. Nếu chỉ dùng cách ước lượng để tính chi phí API cho tiếng Việt thì mình sẽ dự toán thiếu, vì số token thực tế thường lớn hơn do cách tách token của mô hình không giống cách đếm số từ.*

---

## Block 3 — Streaming & Độ Bền (trả lời sau Checkpoint 3)

### Câu 3.1 — Trải nghiệm người dùng với streaming
**Xét ba ứng dụng: (a) chatbot văn bản, (b) trợ lý giọng nói đọc to phản hồi,
(c) pipeline dịch tài liệu chạy ngầm ban đêm. Ứng dụng nào hưởng lợi nhiều
nhất từ streaming, ứng dụng nào không cần — và tại sao?** (1 đoạn văn)
> *Theo mình, chatbot văn bản và đặc biệt là trợ lý giọng nói hưởng lợi nhiều nhất từ streaming vì người dùng có thể thấy hoặc nghe phản hồi ngay khi AI đang sinh nội dung, tạo cảm giác nhanh và tự nhiên hơn. Ngược lại, pipeline dịch tài liệu chạy ngầm ban đêm hầu như không cần streaming vì không có người dùng chờ kết quả trực tiếp, chỉ cần nhận bản dịch hoàn chỉnh khi xử lý xong.*

### Câu 3.2 — Vì sao backoff theo cấp số nhân?
**Khi API quá tải và hàng nghìn client cùng retry, exponential backoff giúp
gì so với delay cố định? Tra cứu thêm: kỹ thuật "jitter" (thêm độ trễ ngẫu
nhiên) giải quyết vấn đề gì còn sót lại?**
> *Exponential backoff giúp các client giãn thời gian retry khi API quá tải, tránh việc tất cả cùng gửi yêu cầu lại một lúc như khi dùng delay cố định, từ đó giảm áp lực lên server và tăng khả năng phục hồi. Tuy nhiên, nhiều client vẫn có thể retry cùng thời điểm, nên người ta thêm jitter (độ trễ ngẫu nhiên) để phân tán thời gian gửi yêu cầu, tránh hiện tượng nhiều client đồng loạt retry cùng lúc*

---

## Block 4 — Mini-Project (trả lời sau Checkpoint 4)

### Câu 4.1 — Thiết kế persona
**Viết lại system prompt bạn dùng cho trợ lý của mình. Chỉ ra 2 chỗ trong
prompt mà nếu xóa đi, hành vi trợ lý sẽ thay đổi rõ rệt — và mô tả thay đổi
đó:**
> *System prompt: "Bạn là trợ lý học tập về lập trình. Hãy trả lời ngắn gọn, dễ hiểu, giải thích từng bước và luôn đưa ví dụ minh họa khi cần."Nếu bỏ phần "trả lời ngắn gọn" thì AI sẽ có xu hướng trả lời dài và lan man hơn. Nếu bỏ phần "luôn đưa ví dụ minh họa khi cần" thì câu trả lời sẽ ít ví dụ thực tế hơn, khiến người đọc khó hình dung và áp dụng*

### Câu 4.2 — Hạn chế & cải thiện
**Trợ lý của bạn giữ history 4 lượt cuối. Hãy mô tả một tình huống hội thoại
cụ thể mà giới hạn này khiến trợ lý trả lời sai/mất ngữ cảnh, và đề xuất một
cách khắc phục (ví dụ: tóm tắt các lượt cũ, tăng giới hạn có chọn lọc...):**
> *Nếu người dùng hỏi về một dự án trong nhiều lượt hội thoại rồi sau vài câu hỏi khác mới quay lại nhắc "tiếp tục phần trước", trợ lý có thể quên nội dung ban đầu vì các tin nhắn cũ đã bị xóa khỏi history. Theo mình, cách khắc phục là tóm tắt ngắn các lượt hội thoại cũ và giữ lại bản tóm tắt, hoặc chỉ giữ những thông tin quan trọng thay vì xóa hoàn toàn khi vượt quá giới hạn 4 lượt.*

---

## Danh Sách Kiểm Tra Nộp Bài

- [ ] `python grade.py` — xem điểm tự động, mục tiêu ≥ 75/100
- [ ] Cả 4 checkpoint pytest đều pass
- [ ] Tất cả 9 câu trong file này đã được trả lời
- [ ] Đã copy bài làm vào folder `solution/`, push lên GitHub cá nhân và nộp link repo vào vlearn (theo hướng dẫn README)
