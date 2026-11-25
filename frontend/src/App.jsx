import { useEffect, useRef, useState } from "react";
import "./App.css";

const API_URL = "http://127.0.0.1:8000";

function renderInlineMarkdown(text) {
  const normalized = text.replace(/\\\*/g, "*").replace(/\\_/g, "_");
  return normalized.split(/(\*\*.*?\*\*|\*[^*]+\*)/g).map((part, index) => {
    if (part.startsWith("**") && part.endsWith("**")) return <strong key={index}>{part.slice(2, -2)}</strong>;
    if (part.startsWith("*") && part.endsWith("*")) return <em key={index}>{part.slice(1, -1)}</em>;
    return part;
  });
}

function renderChatContent(content) {
  const normalized = content.replace(/\\\*/g, "*").replace(/\\n/g, "\n").trim();
  const lines = normalized.split("\n");
  const elements = [];
  let paragraph = [];
  let listItems = [];
  const flushParagraph = () => {
    if (paragraph.length) {
      elements.push(<p key={`p-${elements.length}`}>{renderInlineMarkdown(paragraph.join(" "))}</p>);
      paragraph = [];
    }
  };
  const flushList = () => {
    if (listItems.length) {
      elements.push(<ul key={`ul-${elements.length}`}>{listItems.map((item, index) => <li key={index}>{renderInlineMarkdown(item)}</li>)}</ul>);
      listItems = [];
    }
  };
  lines.forEach((line) => {
    const trimmed = line.trim();
    if (!trimmed) { flushParagraph(); flushList(); return; }
    const bulletMatch = trimmed.match(/^[-*•]\s+(.+)/);
    if (bulletMatch) { flushParagraph(); listItems.push(bulletMatch[1]); }
    else { flushList(); paragraph.push(trimmed); }
  });
  flushParagraph();
  flushList();
  return elements;
}

// Chỉ dùng để hiển thị: tách nội dung kiến thức thành các ý đánh số
function splitItems(text = "") {
  let items = text.split(/\n+/).map((s) => s.replace(/^[-*•\d.)\s]+/, "").trim()).filter(Boolean);
  if (items.length === 1 && items[0].length > 160) {
    items = items[0].split(/(?<=[.!?])\s+/).map((s) => s.trim()).filter(Boolean);
  }
  return items;
}

const ICONS = {
  camera: <><path d="M14.5 4h-5L7 7H4a2 2 0 0 0-2 2v9a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2V9a2 2 0 0 0-2-2h-3l-2.5-3z" /><circle cx="12" cy="13" r="3" /></>,
  leaf: <><path d="M11 20A7 7 0 0 1 9.8 6.1C15.5 5 17 4.48 19 2c1 2 2 4.18 2 8 0 5.5-4.78 10-10 10Z" /><path d="M2 21c0-3 1.85-5.36 5.08-6C9.5 14.52 12 13 13 12" /></>,
  search: <><circle cx="11" cy="11" r="8" /><path d="m21 21-4.3-4.3" /></>,
  bug: <><path d="m8 2 1.88 1.88" /><path d="M14.12 3.88 16 2" /><path d="M9 7.13v-1a3.003 3.003 0 1 1 6 0v1" /><path d="M12 20c-3.3 0-6-2.7-6-6v-3a4 4 0 0 1 4-4h4a4 4 0 0 1 4 4v3c0 3.3-2.7 6-6 6" /><path d="M12 20v-9" /><path d="M6.53 9C4.6 8.8 3 7.1 3 5" /><path d="M6 13H2" /><path d="M3 21c0-2.1 1.7-3.9 3.8-4" /><path d="M20.97 5c0 2.1-1.6 3.8-3.5 4" /><path d="M22 13h-4" /><path d="M17.2 17c2.1.1 3.8 1.9 3.8 4" /></>,
  shield: <><path d="M20 13c0 5-3.5 7.5-7.66 8.95a1 1 0 0 1-.67-.01C7.5 20.5 4 18 4 13V6a1 1 0 0 1 1-1c2 0 4.5-1.2 6.24-2.72a1.17 1.17 0 0 1 1.52 0C14.51 3.81 17 5 19 5a1 1 0 0 1 1 1z" /><path d="m9 12 2 2 4-4" /></>,
  sprout: <><path d="M7 20h10" /><path d="M10 20c5.5-2.5.8-6.4 3-10" /><path d="M9.5 9.4c1.1.8 1.8 2.2 2.3 3.7-2 .4-3.5.4-4.8-.3-1.2-.6-2.3-1.9-3-4.2 2.8-.5 4.4 0 5.5.8z" /><path d="M14.1 6a7 7 0 0 0-1.1 4c1.9-.1 3.3-.6 4.3-1.4 1-1 1.6-2.3 1.7-4.6-2.7.1-4 1-4.9 2z" /></>,
  book: <path d="M4 19.5v-15A2.5 2.5 0 0 1 6.5 2H19a1 1 0 0 1 1 1v18a1 1 0 0 1-1 1H6.5a1 1 0 0 1 0-5H20" />,
  message: <path d="M7.9 20A9 9 0 1 0 4 16.1L2 22Z" />,
  zap: <path d="M4 14a1 1 0 0 1-.78-1.63l9.9-10.2a.5.5 0 0 1 .86.46l-1.92 6.02A1 1 0 0 0 13 10h7a1 1 0 0 1 .78 1.63l-9.9 10.2a.5.5 0 0 1-.86-.46l1.92-6.02A1 1 0 0 0 11 14z" />,
  lightbulb: <><path d="M15 14c.2-1 .7-1.7 1.5-2.5 1-.9 1.5-2.2 1.5-3.5A6 6 0 0 0 6 8c0 1 .2 2.2 1.5 3.5.7.7 1.3 1.5 1.5 2.5" /><path d="M9 18h6" /><path d="M10 22h4" /></>,
  lock: <><rect width="18" height="11" x="3" y="11" rx="2" /><path d="M7 11V7a5 5 0 0 1 10 0v4" /></>,
  check: <path d="M20 6 9 17l-5-5" />,
  info: <><circle cx="12" cy="12" r="10" /><path d="M12 16v-4" /><path d="M12 8h.01" /></>,
  alert: <><circle cx="12" cy="12" r="10" /><path d="M12 8v5" /><path d="M12 17h.01" /></>,
  x: <><path d="M18 6 6 18" /><path d="m6 6 12 12" /></>,
  send: <><path d="M14.536 21.686a.5.5 0 0 0 .937-.024l6.5-19a.496.496 0 0 0-.635-.635l-19 6.5a.5.5 0 0 0-.024.937l7.93 3.18a2 2 0 0 1 1.112 1.11z" /><path d="m21.854 2.147-10.94 10.939" /></>,
  upload: <><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" /><path d="m17 8-5-5-5 5" /><path d="M12 3v12" /></>,
  sparkles: <path d="M9.937 15.5A2 2 0 0 0 8.5 14.063l-6.135-1.582a.5.5 0 0 1 0-.962L8.5 9.936A2 2 0 0 0 9.937 8.5l1.582-6.135a.5.5 0 0 1 .963 0L14.063 8.5A2 2 0 0 0 15.5 9.937l6.135 1.581a.5.5 0 0 1 0 .964L15.5 14.063a2 2 0 0 0-1.437 1.437l-1.582 6.135a.5.5 0 0 1-.963 0z" />,
};

function Icon({ name, size = 20 }) {
  return (
    <svg className="icon" width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">{ICONS[name]}</svg>
  );
}

const KNOWLEDGE_SECTIONS = {
  symptoms: { title: "Triệu chứng", icon: "search", description: "Các dấu hiệu nhận biết bệnh trên cây trồng." },
  causes: { title: "Nguyên nhân", icon: "bug", description: "Tác nhân và điều kiện gây ra bệnh." },
  prevention: { title: "Phòng ngừa", icon: "shield", description: "Các biện pháp hạn chế nguy cơ mắc bệnh." },
  treatment: { title: "Cách điều trị", icon: "sprout", description: "Hướng dẫn xử lý và chăm sóc cây trồng." },
};

function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [activeKnowledgeTab, setActiveKnowledgeTab] = useState("symptoms");
  const [chatInput, setChatInput] = useState("");
  const [chatMessages, setChatMessages] = useState([]);
  const [chatLoading, setChatLoading] = useState(false);
  const [chatError, setChatError] = useState("");
  const [chatOpen, setChatOpen] = useState(false);
  const chatBoxRef = useRef(null);
  const chatWidgetRef = useRef(null);
  const dragRef = useRef(null);
  const [chatPos, setChatPos] = useState(null);
  const [chatSize, setChatSize] = useState(null);
  const resizeRef = useRef(null);

  useEffect(() => {
    const clamp = () => {
      setChatSize((size) => size && {
        w: Math.min(size.w, window.innerWidth),
        h: Math.min(size.h, window.innerHeight),
      });
      setChatPos((pos) => {
        const el = chatWidgetRef.current;
        if (!pos || !el) return pos;
        return {
          left: Math.min(Math.max(0, pos.left), Math.max(0, window.innerWidth - el.offsetWidth)),
          top: Math.min(Math.max(0, pos.top), Math.max(0, window.innerHeight - el.offsetHeight)),
        };
      });
    };
    window.addEventListener("resize", clamp);
    return () => window.removeEventListener("resize", clamp);
  }, []);

  const handleDragStart = (event) => {
    if (event.target.closest("button") || !chatWidgetRef.current) return;
    const rect = chatWidgetRef.current.getBoundingClientRect();
    dragRef.current = { dx: event.clientX - rect.left, dy: event.clientY - rect.top, w: rect.width, h: rect.height };
    event.currentTarget.setPointerCapture(event.pointerId);
    setChatPos({ left: rect.left, top: rect.top });
  };
  const handleDragMove = (event) => {
    const d = dragRef.current;
    if (!d) return;
    setChatPos({
      left: Math.min(Math.max(0, event.clientX - d.dx), Math.max(0, window.innerWidth - d.w)),
      top: Math.min(Math.max(0, event.clientY - d.dy), Math.max(0, window.innerHeight - d.h)),
    });
  };
  const handleDragEnd = () => { dragRef.current = null; };

  const MIN_CHAT_W = 320;
  const MIN_CHAT_H = 380;

  const handleResizeStart = (dir) => (event) => {
    if (!chatWidgetRef.current) return;
    event.preventDefault();
    event.stopPropagation();
    const rect = chatWidgetRef.current.getBoundingClientRect();
    resizeRef.current = { dir, x: event.clientX, y: event.clientY, left: rect.left, top: rect.top, w: rect.width, h: rect.height };
    event.currentTarget.setPointerCapture(event.pointerId);
    setChatPos({ left: rect.left, top: rect.top });
    setChatSize({ w: rect.width, h: rect.height });
  };
  const handleResizeMove = (event) => {
    const r = resizeRef.current;
    if (!r) return;
    const dx = event.clientX - r.x;
    const dy = event.clientY - r.y;
    const right = r.left + r.w;
    const bottom = r.top + r.h;
    let { left, top, w, h } = r;
    if (r.dir.includes("e")) w = Math.min(Math.max(MIN_CHAT_W, r.w + dx), window.innerWidth - r.left);
    if (r.dir.includes("w")) { w = Math.min(Math.max(MIN_CHAT_W, r.w - dx), right); left = right - w; }
    if (r.dir.includes("s")) h = Math.min(Math.max(MIN_CHAT_H, r.h + dy), window.innerHeight - r.top);
    if (r.dir.includes("n")) { h = Math.min(Math.max(MIN_CHAT_H, r.h - dy), bottom); top = bottom - h; }
    setChatPos({ left, top });
    setChatSize({ w, h });
  };
  const handleResizeEnd = () => { resizeRef.current = null; };

  useEffect(() => {
    const el = chatBoxRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [chatMessages, chatLoading, chatOpen]);

  useEffect(() => () => { if (preview) URL.revokeObjectURL(preview); }, [preview]);

  const handleFileChange = (event) => {
    const selectedFile = event.target.files?.[0];
    if (!selectedFile) return;
    if (!selectedFile.type.startsWith("image/")) {
      setError("Vui lòng chọn một tệp hình ảnh hợp lệ."); event.target.value = ""; return;
    }
    if (selectedFile.size > 10 * 1024 * 1024) {
      setError("Kích thước ảnh không được vượt quá 10 MB."); event.target.value = ""; return;
    }
    setFile(selectedFile);
    setPreview(URL.createObjectURL(selectedFile));
    setResult(null);
    setActiveKnowledgeTab("symptoms");
    setError("");
  };

  const handlePredict = async () => {
    if (!file) { setError("Vui lòng tải ảnh lá cây lên trước."); return; }
    setLoading(true); setError(""); setResult(null);
    try {
      const formData = new FormData();
      formData.append("file", file);
      const response = await fetch(`${API_URL}/predict`, { method: "POST", body: formData });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || data.message || "Không thể chẩn đoán ảnh.");
      if (!data.success) throw new Error(data.message || "Chẩn đoán thất bại.");
      setResult(data.data);
      setActiveKnowledgeTab("symptoms");
    } catch (err) {
      setError(err.message || "Không thể kết nối đến máy chủ. Hãy kiểm tra backend FastAPI.");
    } finally { setLoading(false); }
  };

  const handleSendMessage = async (event) => {
    event.preventDefault();
    const question = chatInput.trim();
    if (!question || chatLoading) return;
    setChatMessages((previous) => [...previous, { role: "user", content: question }]);
    setChatInput(""); setChatError(""); setChatLoading(true);
    try {
      const prediction = result?.prediction;
      const response = await fetch(`${API_URL}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question,
          plant: prediction?.top_prediction?.plant || null,
          disease: prediction?.top_prediction?.disease || null,
          top_k: 8,
        }),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "Không thể gửi câu hỏi đến hệ thống.");
      if (!data.success) throw new Error("AI không thể tạo câu trả lời.");
      setChatMessages((previous) => [...previous, { role: "assistant", content: data.answer, knowledge: data.knowledge }]);
    } catch (err) {
      setChatError(err.message || "Không thể kết nối đến máy chủ.");
    } finally { setChatLoading(false); }
  };

  const handleReset = () => {
    setFile(null); setPreview(""); setResult(null); setError("");
    setChatError(""); setChatMessages([]); setChatInput(""); setActiveKnowledgeTab("symptoms");
    const input = document.getElementById("image-upload");
    if (input) input.value = "";
  };

  const prediction = result?.prediction;
  const knowledge = result?.knowledge;
  const sections = knowledge?.sections || [];
  const availableSections = Object.keys(KNOWLEDGE_SECTIONS).map((key) => ({
    key,
    config: KNOWLEDGE_SECTIONS[key],
    section: sections.find((item) => item.section === key),
  }));
  const activeSection = availableSections.find((item) => item.key === activeKnowledgeTab) || availableSections[0];
  const items = splitItems(activeSection?.section?.content);
  const hasResult = Boolean(result && prediction?.top_prediction);

  return (
    <div className="app">
      <header className="header">
        <a className="brand" href="#home" aria-label="PlantCare AI - Trang chủ"><div className="brand-icon"><Icon name="leaf" size={24} /></div><div><h1>PlantCare AI</h1><p>Hệ thống chẩn đoán bệnh cây trồng</p></div></a>
        <div className="header-right"><span className="pill"><Icon name="sparkles" size={14} /> AI-Powered</span><div className="status"><span className="status-dot" />Hệ thống chẩn đoán</div></div>
      </header>

      <div className="shell">
        <aside className="sidebar">
          <nav aria-label="Điều hướng">
            <a className="nav-item active" href="#diagnosis"><Icon name="camera" size={19} />Chẩn đoán</a>
            <a className="nav-item" href="#knowledge"><Icon name="book" size={19} />Kiến thức nông nghiệp</a>
            <button type="button" className={`nav-item ${chatOpen ? "is-open" : ""}`} onClick={() => setChatOpen((open) => !open)}><Icon name="message" size={19} />Trợ lý AI</button>
          </nav>
          <div className="side-card"><strong>MobileNetV2 + RAG</strong><p>Nhận diện bệnh từ ảnh lá, truy xuất kiến thức và tư vấn cùng Gemini AI.</p></div>
        </aside>

        <main className="content" id="home">
          <section className="page-head">
            <h2><span className="page-icon"><Icon name="leaf" size={28} /></span>Chẩn đoán bệnh cây trồng bằng AI</h2>
            <p>Tải ảnh lá cây để nhận diện bệnh tức thì, kèm kiến thức và hướng xử lý phù hợp.</p>
          </section>

          <section className={`workspace-grid ${hasResult || loading ? "has-result" : ""}`} id="diagnosis">
            <div className="left-col">
              <section className="upload-card">
                <input id="image-upload" type="file" accept="image/jpeg,image/png,image/webp" onChange={handleFileChange} hidden />
                <label className={`upload-area ${preview ? "has-preview" : ""}`} htmlFor="image-upload">
                  {preview ? <><img src={preview} alt="Ảnh lá cây được tải lên" /><span className="upload-hint">Nhấn để đổi ảnh</span></> : <><div className="upload-icon"><Icon name="upload" size={30} /></div><strong>Chọn ảnh lá cây để tải lên</strong><span>Nhấn vào khu vực này để tải ảnh từ thiết bị</span><small>JPG, PNG, WEBP · Tối đa 10 MB</small></>}
                </label>
                {file && <p className="file-name"><span className="file-check"><Icon name="check" size={14} /></span>Tệp đã chọn: <strong>{file.name}</strong></p>}
                {error && <div className="error-message">{error}</div>}
                <div className="action-buttons">
                  <label className="btn btn-green" htmlFor="image-upload"><Icon name="camera" size={18} /> Chọn ảnh</label>
                  <button className="btn btn-blue" onClick={handlePredict} disabled={!file || loading} type="button">{loading ? <><span className="button-spinner" /> Đang phân tích...</> : <><Icon name="zap" size={18} /> Chẩn đoán bằng AI</>}</button>
                  {file && <button className="btn btn-ghost" onClick={handleReset} type="button" disabled={loading}>Làm mới</button>}
                </div>
              </section>
              {!hasResult && <section className="tips-card">
                <h4><Icon name="lightbulb" size={18} /> Mẹo chụp ảnh để có kết quả tốt nhất</h4>
                <ul><li>Chụp rõ một chiếc lá có dấu hiệu bệnh, đủ ánh sáng tự nhiên.</li><li>Giữ ảnh nét, lá chiếm phần lớn khung hình.</li><li>Tránh bóng đổ và nền quá rối.</li></ul>
                <p><Icon name="lock" size={13} /> Ảnh được gửi đến hệ thống để phân tích và trả kết quả.</p>
              </section>}
            </div>

          {(loading || hasResult) && <section className="result-card" id="result" aria-live="polite">
            {loading ? <div className="result-state"><div className="spinner" /><h3>Đang nhận diện tình trạng lá cây</h3><p>Hệ thống đang xử lý hình ảnh và truy xuất kiến thức liên quan.</p></div> : <>
              <div className="result-main">
                <div>
                  <div className="result-top"><span className="result-badge"><Icon name="alert" size={30} /></span><div><h3>{prediction.top_prediction.plant} – {prediction.top_prediction.disease}</h3><span className="sub">Phân tích AI hoàn tất</span></div></div>
                  <div className="confidence-row"><span>Độ tin cậy</span><strong>{prediction.top_prediction.confidence}%</strong></div>
                  <div className="progress big"><div className="progress-fill" style={{ width: `${prediction.top_prediction.confidence}%` }} /></div>
                </div>
                <div className="top-list"><h4>Top 3 kết quả dự đoán</h4>{prediction.top_predictions.map((item) => <div className="prediction-item" key={item.class_id}><span className="rank">#{item.rank}</span><div className="prediction-name"><strong>{item.disease}</strong><span>{item.plant}</span></div><div className="prediction-score"><strong>{item.confidence}%</strong><div className="progress"><div className="progress-fill" style={{ width: `${item.confidence}%` }} /></div></div></div>)}</div>
              </div>
              <div className="disclaimer"><Icon name="info" size={17} /><p>Kết quả do AI dự đoán chỉ mang tính tham khảo, không thay thế việc kiểm tra và tư vấn trực tiếp từ chuyên gia nông nghiệp. Mô hình: {prediction.model}</p></div>
            </>}
          </section>}
          </section>

          {hasResult && <div className="knowledge-section" id="knowledge"><div className="section-title"><span className="title-icon"><Icon name="book" size={22} /></span><div><h2>Kiến thức và tư vấn nông nghiệp</h2><p>Thông tin truy xuất từ cơ sở dữ liệu nông nghiệp của hệ thống.</p></div></div><section className="knowledge-card">
              <div className="knowledge-head"><span className="step-label">Bệnh được chẩn đoán</span><h3>{prediction.top_prediction.plant} – {prediction.top_prediction.disease}</h3></div>
              {knowledge?.available && sections.length > 0 ? <>
                <div className="knowledge-tabs" role="tablist" aria-label="Danh mục kiến thức bệnh cây trồng">
                  {availableSections.map(({ key, config, section }) => <button key={key} type="button" role="tab" aria-selected={activeKnowledgeTab === key} aria-controls={`knowledge-panel-${key}`} className={`knowledge-tab ${activeKnowledgeTab === key ? "active" : ""} ${!section ? "unavailable" : ""}`} onClick={() => setActiveKnowledgeTab(key)}><Icon name={config.icon} size={18} />{config.title}</button>)}
                </div>
                {activeSection?.section ? <article className="knowledge-detail" id={`knowledge-panel-${activeSection.key}`} role="tabpanel">
                  <ol className="step-list">{items.map((text, i) => <li key={i}><span className="num">{i + 1}</span><span>{text}</span></li>)}</ol>
                  {activeSection.section.source && <div className="knowledge-source"><strong>Nguồn tham khảo:</strong> {activeSection.section.source.startsWith("http") ? <a href={activeSection.section.source} target="_blank" rel="noopener noreferrer">{activeSection.section.source}</a> : <span>{activeSection.section.source}</span>}</div>}
                </article> : <div className="knowledge-empty"><strong>Chưa có nội dung cho mục này</strong><p>Cơ sở dữ liệu chưa trả về thông tin cho mục {activeSection?.config.title.toLowerCase()}. Hãy chọn tab khác.</p></div>}
              </> : <div className="knowledge-empty"><strong>{knowledge?.message || "Chưa có kiến thức cho kết quả này"}</strong><p>{knowledge?.message ? "Bạn có thể hỏi trợ lý Gemini AI ở bên dưới." : "Nội dung triệu chứng, nguyên nhân, phòng ngừa và điều trị sẽ hiển thị tại đây."}</p></div>}
            </section></div>}

          {chatOpen && <section className="chat-card chat-widget" id="assistant" role="dialog" aria-label="Trợ lý AI" ref={chatWidgetRef} style={chatPos ? { left: chatPos.left, top: chatPos.top, right: "auto", bottom: "auto", ...(chatSize ? { width: chatSize.w, height: chatSize.h } : {}) } : undefined}>
            {["nw", "ne", "sw", "se"].map((dir) => <span key={dir} className={`chat-resize chat-resize-${dir}`} onPointerDown={handleResizeStart(dir)} onPointerMove={handleResizeMove} onPointerUp={handleResizeEnd} onPointerCancel={handleResizeEnd} aria-hidden="true" />)}
            <div className="chat-heading" onPointerDown={handleDragStart} onPointerMove={handleDragMove} onPointerUp={handleDragEnd} onPointerCancel={handleDragEnd}><div className="chat-title-group"><div className="chat-avatar"><Icon name="sparkles" size={20} /></div><div><h3>Trợ lý tư vấn cây trồng</h3><p>Trao đổi cùng Gemini AI về bệnh cây trồng và cách chăm sóc.</p></div></div><span className="gemini-tag"><Icon name="sparkles" size={13} /> GEMINI AI</span><button type="button" className="chat-close" onClick={() => setChatOpen(false)} aria-label="Đóng trợ lý AI"><Icon name="x" size={18} /></button></div>
            {prediction?.top_prediction && <div className="chat-context"><span className="context-indicator" /><span>Ngữ cảnh chẩn đoán:<strong> {prediction.top_prediction.plant} – {prediction.top_prediction.disease}</strong></span></div>}
            <div className="chat-messages" ref={chatBoxRef}>
              {chatMessages.length === 0 && <div className="chat-empty"><div className="chat-empty-icon"><Icon name="leaf" size={26} /></div><h4>Xin chào! Mình là PlantCare AI</h4><p>Mình có thể hỗ trợ bạn tìm hiểu triệu chứng, nguyên nhân, cách phòng ngừa và chăm sóc cây trồng.</p><div className="chat-suggestions"><button type="button" onClick={() => setChatInput("Làm thế nào để chăm sóc cây cà chua khỏe mạnh?")}><Icon name="sprout" size={15} /> Cách chăm sóc cà chua</button><button type="button" onClick={() => setChatInput("Bệnh Early blight trên cà chua có triệu chứng gì?")}><Icon name="search" size={15} /> Triệu chứng Early blight</button></div></div>}
              {chatMessages.map((message, index) => <div className={`chat-message ${message.role}`} key={index}><div className="chat-message-label">{message.role === "user" ? "Bạn" : "PlantCare AI"}</div><div className="chat-message-content">{renderChatContent(message.content)}</div>{message.knowledge?.sources?.length > 0 && <div className="chat-sources"><strong>Nguồn kiến thức</strong>{message.knowledge.sources.map((source, sourceIndex) => <div className="chat-source-item" key={sourceIndex}><span>{source.plant} – {source.disease}: </span>{source.source.startsWith("http") ? <a href={source.source} target="_blank" rel="noopener noreferrer">{source.source.includes("extension.umn.edu") ? "University of Minnesota Extension – Early blight" : source.source}</a> : <span>{source.source}</span>}</div>)}</div>}</div>)}
              {chatLoading && <div className="chat-loading"><span className="typing-dots"><i /><i /><i /></span>PlantCare AI đang suy nghĩ...</div>}
            </div>
            {chatError && <div className="error-message">{chatError}</div>}
            <form className="chat-form" onSubmit={handleSendMessage}><input type="text" value={chatInput} onChange={(event) => setChatInput(event.target.value)} placeholder="Nhập câu hỏi về cây trồng..." disabled={chatLoading} aria-label="Nhập câu hỏi về cây trồng" /><button type="submit" disabled={!chatInput.trim() || chatLoading}>{chatLoading ? "Đang gửi..." : <>Gửi <Icon name="send" size={15} /></>}</button></form>
            <p className="chat-note">Câu trả lời được tạo bởi Gemini AI dựa trên kiến thức có trong hệ thống PlantCare AI.</p>
          </section>}
        </main>
        {!chatOpen && <button type="button" className="chat-fab" onClick={() => setChatOpen(true)} aria-label="Mở trợ lý AI"><Icon name="message" size={26} /></button>}
      </div>

      {/* ===== FOOTER ===== */}
      <footer className="footer">
        <div className="footer-inner">
          <div className="footer-col footer-about">
            <div className="footer-brand">
              <span className="footer-brand-icon"><Icon name="leaf" size={22} /></span>
              <strong>PlantCare AI</strong>
            </div>
            <p>Hệ thống chẩn đoán bệnh cây trồng từ ảnh lá, kết hợp truy xuất kiến thức nông nghiệp và tư vấn cùng Gemini AI.</p>
            <span className="footer-badge"><Icon name="sparkles" size={13} /> AI-Powered</span>
          </div>

          <div className="footer-col">
            <h4>Điều hướng</h4>
            <ul className="footer-links">
              <li><a href="#home"><Icon name="leaf" size={15} />Trang chủ</a></li>
              <li><a href="#diagnosis"><Icon name="camera" size={15} />Chẩn đoán</a></li>
              <li><a href="#knowledge"><Icon name="book" size={15} />Kiến thức nông nghiệp</a></li>
              <li><button type="button" onClick={() => setChatOpen(true)}><Icon name="message" size={15} />Trợ lý AI</button></li>
            </ul>
          </div>

          <div className="footer-col">
            <h4>Công nghệ</h4>
            <div className="footer-tags">
              <span>MobileNetV2</span>
              <span>RAG</span>
              <span>Gemini AI</span>
              <span>FastAPI</span>
              <span>React</span>
            </div>
            <p className="footer-note"><Icon name="lock" size={14} />Ảnh chỉ được dùng để phân tích và trả kết quả chẩn đoán.</p>
          </div>

          <div className="footer-col">
            <h4>Lưu ý</h4>
            <p className="footer-warn"><Icon name="info" size={16} />Kết quả do AI dự đoán chỉ mang tính tham khảo, không thay thế tư vấn trực tiếp từ chuyên gia nông nghiệp.</p>
          </div>
        </div>

        <div className="footer-bottom">
          <span>© {new Date().getFullYear()} PlantCare AI. Bảo lưu mọi quyền.</span>
          <span className="footer-bottom-right"><Icon name="shield" size={14} />Chăm sóc cây trồng thông minh</span>
        </div>
      </footer>
    </div>
  );
}

export default App;