import { useEffect, useState } from "react";
import "./App.css";

const API_URL = "http://127.0.0.1:8000";

// ==========================================
// CHAT MESSAGE FORMATTER
// ==========================================

function renderInlineMarkdown(text) {
  const normalized = text
    .replace(/\\\*/g, "*")
    .replace(/\\_/g, "_");

  return normalized.split(/(\*\*.*?\*\*|\*[^*]+\*)/g).map((part, index) => {
    if (part.startsWith("**") && part.endsWith("**")) {
      return <strong key={index}>{part.slice(2, -2)}</strong>;
    }

    if (part.startsWith("*") && part.endsWith("*")) {
      return <em key={index}>{part.slice(1, -1)}</em>;
    }

    return part;
  });
}

function renderChatContent(content) {
  const normalized = content
    .replace(/\\\*/g, "*")
    .replace(/\\n/g, "\n")
    .trim();

  const lines = normalized.split("\n");
  const elements = [];
  let paragraph = [];
  let listItems = [];

  const flushParagraph = () => {
    if (paragraph.length > 0) {
      elements.push(
        <p key={`p-${elements.length}`}>
          {renderInlineMarkdown(paragraph.join(" "))}
        </p>
      );
      paragraph = [];
    }
  };

  const flushList = () => {
    if (listItems.length > 0) {
      elements.push(
        <ul key={`ul-${elements.length}`}>
          {listItems.map((item, index) => (
            <li key={index}>
              {renderInlineMarkdown(item)}
            </li>
          ))}
        </ul>
      );
      listItems = [];
    }
  };

  lines.forEach((line) => {
    const trimmed = line.trim();

    if (!trimmed) {
      flushParagraph();
      flushList();
      return;
    }

    const bulletMatch = trimmed.match(/^[-*•]\s+(.+)/);

    if (bulletMatch) {
      flushParagraph();
      listItems.push(bulletMatch[1]);
    } else {
      flushList();
      paragraph.push(trimmed);
    }
  });

  flushParagraph();
  flushList();

  return elements;
}

const KNOWLEDGE_SECTIONS = {
  symptoms: {
    title: "Triệu chứng",
    icon: "🔍",
    description: "Các dấu hiệu nhận biết bệnh trên cây trồng.",
  },
  causes: {
    title: "Nguyên nhân",
    icon: "🦠",
    description: "Tác nhân và điều kiện gây ra bệnh.",
  },
  prevention: {
    title: "Phòng ngừa",
    icon: "🛡️",
    description: "Các biện pháp hạn chế nguy cơ mắc bệnh.",
  },
  treatment: {
    title: "Điều trị",
    icon: "🌱",
    description: "Hướng dẫn xử lý và chăm sóc cây trồng.",
  },
};

function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  // Chatbot states
  const [chatInput, setChatInput] = useState("");
  const [chatMessages, setChatMessages] = useState([]);
  const [chatLoading, setChatLoading] = useState(false);
  const [chatError, setChatError] = useState("");

  useEffect(() => {
    return () => {
      if (preview) URL.revokeObjectURL(preview);
    };
  }, [preview]);

  const handleFileChange = (event) => {
    const selectedFile = event.target.files?.[0];

    if (!selectedFile) return;

    if (!selectedFile.type.startsWith("image/")) {
      setError("Vui lòng chọn một tệp hình ảnh hợp lệ.");
      return;
    }

    if (selectedFile.size > 10 * 1024 * 1024) {
      setError("Kích thước ảnh không được vượt quá 10 MB.");
      event.target.value = "";
      return;
    }

    setFile(selectedFile);
    setPreview(URL.createObjectURL(selectedFile));
    setResult(null);
    setError("");
  };

  const handlePredict = async () => {
    if (!file) {
      setError("Vui lòng tải ảnh lá cây lên trước.");
      return;
    }

    setLoading(true);
    setError("");
    setResult(null);

    try {
      const formData = new FormData();
      formData.append("file", file);

      const response = await fetch(`${API_URL}/predict`, {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          data.detail || data.message || "Không thể chẩn đoán ảnh."
        );
      }

      if (!data.success) {
        throw new Error(data.message || "Chẩn đoán thất bại.");
      }

      setResult(data.data);
    } catch (err) {
      setError(
        err.message ||
          "Không thể kết nối đến máy chủ. Hãy kiểm tra backend FastAPI."
      );
    } finally {
      setLoading(false);
    }
  };

const handleSendMessage = async (event) => {
  event.preventDefault();

  const question = chatInput.trim();

  if (!question || chatLoading) return;

  // Add user's message
  setChatMessages((previous) => [
    ...previous,
    {
      role: "user",
      content: question,
    },
  ]);

  setChatInput("");
  setChatError("");
  setChatLoading(true);

  try {
    const response = await fetch(`${API_URL}/chat`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        question: question,
        plant: prediction?.top_prediction?.plant || null,
        disease: prediction?.top_prediction?.disease || null,
        top_k: 8,
      }),
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(
        data.detail || "Không thể gửi câu hỏi đến hệ thống."
      );
    }

    if (!data.success) {
      throw new Error("AI không thể tạo câu trả lời.");
    }

    setChatMessages((previous) => [
      ...previous,
      {
        role: "assistant",
        content: data.answer,
        knowledge: data.knowledge,
      },
    ]);

  } catch (err) {
    setChatError(
      err.message || "Không thể kết nối đến máy chủ."
    );
  } finally {
    setChatLoading(false);
  }
};

  const handleReset = () => {
  // Reset image
  setFile(null);
  setPreview("");

  // Reset prediction result
  setResult(null);

  // Reset errors
  setError("");
  setChatError("");

  // Reset chatbot
  setChatMessages([]);
  setChatInput("");

  // Reset image input
  const input = document.getElementById("image-upload");
  if (input) input.value = "";
};

  const prediction = result?.prediction;
  const knowledge = result?.knowledge;
  const sections = knowledge?.sections || [];

  return (
    <div className="app">
      <header className="header">
        <div className="header-inner">
          <div className="brand">
            <div className="brand-icon">🌿</div>
            <div>
              <h1>PlantCare AI</h1>
              <p>Hệ thống chẩn đoán bệnh cây trồng</p>
            </div>
          </div>

          <div className="status">
            <span className="status-dot"></span>
            Hệ thống chẩn đoán
          </div>
        </div>
      </header>

      <main className="main">
        <section className="hero">
          <span className="hero-label">AI PLANT DISEASE DIAGNOSIS</span>
          <h2>Chăm sóc cây trồng thông minh hơn</h2>
          <p>
            Tải ảnh lá cây lên để hệ thống AI phân tích, nhận diện
            bệnh và cung cấp kiến thức tư vấn chăm sóc cây trồng.
          </p>
        </section>

        <section className="diagnosis-card">
          <div className="section-heading">
            <div>
              <h3>Chẩn đoán bệnh cây trồng</h3>
              <p>Chọn hình ảnh lá cây từ thiết bị của bạn.</p>
            </div>
            <span className="model-tag">MobileNetV2</span>
          </div>

          <div className="upload-area">
            {preview ? (
              <div className="preview-container">
                <img src={preview} alt="Ảnh lá cây được tải lên" />
                <button
                  className="remove-button"
                  onClick={handleReset}
                  type="button"
                >
                  Xóa ảnh
                </button>
              </div>
            ) : (
              <label className="upload-placeholder" htmlFor="image-upload">
                <div className="upload-icon">📷</div>
                <strong>Chọn ảnh lá cây</strong>
                <span>Nhấn vào đây để tải ảnh từ thiết bị</span>
                <small>Hỗ trợ JPG, JPEG, PNG, WEBP (tối đa 10 MB)</small>
              </label>
            )}

            <input
              id="image-upload"
              type="file"
              accept="image/jpeg,image/png,image/webp"
              onChange={handleFileChange}
              hidden
            />
          </div>

          {file && (
            <p className="file-name">
              Tệp đã chọn: <strong>{file.name}</strong>
            </p>
          )}

          {error && <div className="error-message">{error}</div>}

          <div className="action-buttons">
            <button
              className="predict-button"
              onClick={handlePredict}
              disabled={!file || loading}
              type="button"
            >
              {loading ? "Đang phân tích..." : "🔍 Chẩn đoán ngay"}
            </button>

            {file && (
              <button
                className="reset-button"
                onClick={handleReset}
                type="button"
                disabled={loading}
              >
                Làm mới
              </button>
            )}
          </div>
        </section>

        {loading && (
          <div className="loading">
            <div className="spinner"></div>
            <p>
              AI đang phân tích hình ảnh và truy xuất kiến thức,
              vui lòng chờ...
            </p>
          </div>
        )}

        {result && (
          <>
            <section className="result-card">
              <div className="result-heading">
                <div>
                  <span className="result-label">KẾT QUẢ PHÂN TÍCH</span>
                  <h3>Kết quả chẩn đoán</h3>
                </div>
                <span className="success-tag">Hoàn tất</span>
              </div>

              <div className="top-result">
                <div>
                  <span className="result-caption">Cây trồng</span>
                  <h4>{prediction.top_prediction.plant}</h4>
                </div>

                <div>
                  <span className="result-caption">Bệnh dự đoán</span>
                  <h4>{prediction.top_prediction.disease}</h4>
                </div>

                <div>
                  <span className="result-caption">Độ tin cậy</span>
                  <h4 className="confidence">
                    {prediction.top_prediction.confidence}%
                  </h4>
                </div>
              </div>

              <div className="top-list">
                <h4>Top 3 kết quả dự đoán</h4>

                {prediction.top_predictions.map((item) => (
                  <div className="prediction-item" key={item.class_id}>
                    <div className="prediction-info">
                      <span className="rank">#{item.rank}</span>
                      <div>
                        <strong>{item.disease}</strong>
                        <span>{item.plant}</span>
                      </div>
                    </div>

                    <div className="prediction-score">
                      <strong>{item.confidence}%</strong>
                      <div className="progress">
                        <div
                          className="progress-fill"
                          style={{ width: `${item.confidence}%` }}
                        ></div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>

              <p className="model-info">
                Mô hình sử dụng: {prediction.model}
              </p>
            </section>

            <section className="knowledge-card">
              <div className="knowledge-heading">
                <div>
                  <span className="result-label">
                    RETRIEVAL-AUGMENTED GENERATION
                  </span>
                  <h3>Kiến thức và tư vấn nông nghiệp</h3>
                  <p>
                    Thông tin liên quan đến cây {prediction.top_prediction.plant}
                    {" – "}
                    {prediction.top_prediction.disease}.
                  </p>
                </div>
                <span className="knowledge-tag">RAG</span>
              </div>

              {knowledge?.available && sections.length > 0 ? (
                <div className="knowledge-grid">
                  {sections.map((section) => {
                    const config =
                      KNOWLEDGE_SECTIONS[section.section] || {
                        title: section.section,
                        icon: "📖",
                        description: "Thông tin kiến thức liên quan.",
                      };

                    return (
                      <article
                        className="knowledge-section"
                        key={section.section}
                      >
                        <div className="knowledge-section-heading">
                          <span className="knowledge-icon">
                            {config.icon}
                          </span>
                          <div>
                            <h4>{config.title}</h4>
                            <p>{config.description}</p>
                          </div>
                        </div>

                        <p className="knowledge-content">
                          {section.content}
                        </p>

                        {section.source && (
                          <div className="knowledge-source">
                            <strong>Nguồn tham khảo:</strong>{" "}
                            {section.source}
                          </div>
                        )}
                      </article>
                    );
                  })}
                </div>
              ) : (
                <div className="knowledge-unavailable">
                  <span>📚</span>
                  <p>
                    {knowledge?.message ||
                      "Chưa tìm thấy kiến thức phù hợp cho kết quả dự đoán này."}
                  </p>
                </div>
              )}

              <p className="knowledge-note">
                Kiến thức được truy xuất từ cơ sở dữ liệu nông nghiệp
                của hệ thống.
              </p>
            </section>

            <p className="disclaimer">
              Lưu ý: Kết quả do AI dự đoán và thông tin tư vấn chỉ mang
              tính tham khảo, không thay thế việc kiểm tra và tư vấn
              trực tiếp từ chuyên gia nông nghiệp.
            </p>
          </>
        )}
      </main>

<section className="chat-card">
  <div className="chat-heading">
    <div>
      <span className="result-label">AI AGRICULTURAL ASSISTANT</span>
      <h3>Trợ lý tư vấn cây trồng</h3>
      <p>Đặt câu hỏi về bệnh cây trồng và cách chăm sóc.</p>
    </div>
    <span className="knowledge-tag">GEMINI AI</span>
  </div>

  <div className="chat-messages">
    {chatMessages.length === 0 && (
      <div className="chat-empty">
        <span>🌱</span>
        <p>
          Xin chào! Mình là PlantCare AI.
          Bạn muốn tìm hiểu điều gì về cây trồng?
        </p>

        <div className="chat-suggestions">
          <button
            type="button"
            onClick={() =>
              setChatInput("Làm thế nào để chăm sóc cây cà chua khỏe mạnh?")
            }
          >
            Cách chăm sóc cà chua
          </button>

          <button
            type="button"
            onClick={() =>
              setChatInput("Bệnh Early blight trên cà chua có triệu chứng gì?")
            }
          >
            Triệu chứng Early blight
          </button>
        </div>
      </div>
    )}

    {chatMessages.map((message, index) => (
      <div className={`chat-message ${message.role}`} key={index}>
        <div className="chat-message-label">
          {message.role === "user" ? "Bạn" : "PlantCare AI"}
        </div>

        <div className="chat-message-content">
        {renderChatContent(message.content)}
        </div>

        {message.knowledge?.sources?.length > 0 && (
          <div className="chat-sources">
            <strong>Nguồn kiến thức:</strong>

            {message.knowledge.sources.map((source, sourceIndex) => (
              <div key={sourceIndex}>
                <span>{source.plant} – {source.disease}: </span>

                {source.source.startsWith("http") ? (
                  <a
                href={source.source}
                target="_blank"
                rel="noopener noreferrer"
                >
                {source.source.includes("extension.umn.edu")
                ? "University of Minnesota Extension – Early blight"
                 : source.source}
                </a>
                ) : (
                  <span>{source.source}</span>
                )}
              </div>
            ))}
          </div>
        )}
      </div>
    ))}

    {chatLoading && (
      <div className="chat-loading">
        PlantCare AI đang suy nghĩ...
      </div>
    )}
  </div>

  {chatError && (
    <div className="error-message">{chatError}</div>
  )}

  <form className="chat-form" onSubmit={handleSendMessage}>
    <input
      type="text"
      value={chatInput}
      onChange={(event) => setChatInput(event.target.value)}
      placeholder="Nhập câu hỏi về cây trồng..."
      disabled={chatLoading}
    />

    <button
      type="submit"
      disabled={!chatInput.trim() || chatLoading}
    >
      {chatLoading ? "Đang gửi..." : "Gửi"}
    </button>
  </form>

  <p className="chat-note">
    Câu trả lời được tạo bởi AI dựa trên kiến thức có trong cơ sở dữ liệu PlantCare AI.
  </p>
</section>

      <footer className="footer">
        <p>PlantCare AI – Hệ thống hỗ trợ chẩn đoán bệnh cây trồng</p>
      </footer>
    </div>
  );
}

export default App;
