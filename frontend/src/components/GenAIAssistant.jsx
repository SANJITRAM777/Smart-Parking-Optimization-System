import React, { useState, useRef, useEffect } from "react"
import { sendGenAIChat } from "../api"

const EXAMPLE_QUESTIONS = [
  "Where should I park?",
  "Which slot is closest?",
  "Is P3 likely to become occupied?",
  "Show me available parking.",
  "What is an ultrasonic sensor?",
]

export default function GenAIAssistant() {
  const [messages, setMessages] = useState([
    {
      sender: "assistant",
      text: "Hello! I am your Smart Parking Assistant. Ask me where to park, which space is closest, or check any slot's availability.",
    },
  ])
  const [query, setQuery] = useState("")
  const [loading, setLoading] = useState(false)
  const chatScrollRef = useRef(null)

  useEffect(() => {
    if (chatScrollRef.current) {
      chatScrollRef.current.scrollTop = chatScrollRef.current.scrollHeight
    }
  }, [messages, loading])

  const handleSend = async (questionText) => {
    const textToSend = (questionText || query).trim()
    if (!textToSend || loading) return

    const userMsg = { sender: "user", text: textToSend }
    setMessages((prev) => [...prev, userMsg])
    setQuery("")
    setLoading(true)

    try {
      const res = await sendGenAIChat(textToSend)
      const botText = res.response || "Parking Assistant is temporarily unavailable. Please try again."
      setMessages((prev) => [
        ...prev,
        {
          sender: "assistant",
          text: botText,
          isError: res.mode === "unavailable" || res.mode === "error",
        },
      ])
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        {
          sender: "assistant",
          text: "Parking Assistant is temporarily unavailable. Please try again.",
          isError: true,
        },
      ])
    } finally {
      setLoading(false)
    }
  }

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  return (
    <div className="driver-assistant-card">
      <div className="assistant-card-header">
        <div className="assistant-header-title">
          <span className="assistant-avatar-icon">🤖</span>
          <div>
            <h3>Parking Assistant</h3>
            <p className="assistant-subtitle">Real-time answers powered by Gemini AI</p>
          </div>
        </div>
        <div className="assistant-live-badge">
          <span className="live-dot"></span>
          <span>Live Assistant</span>
        </div>
      </div>

      <div className="chat-thread-container" ref={chatScrollRef}>
        {messages.map((m, idx) => (
          <div
            key={idx}
            className={`chat-bubble-row ${m.sender === "user" ? "row-user" : "row-assistant"}`}
          >
            <div className="bubble-avatar">{m.sender === "user" ? "👤" : "🤖"}</div>
            <div className={`chat-bubble ${m.sender === "user" ? "bubble-user" : "bubble-assistant"} ${m.isError ? "bubble-error" : ""}`}>
              <div className="bubble-sender">{m.sender === "user" ? "You" : "Parking Assistant"}</div>
              <div className="bubble-text">{m.text}</div>
            </div>
          </div>
        ))}

        {loading && (
          <div className="chat-bubble-row row-assistant">
            <div className="bubble-avatar">🤖</div>
            <div className="chat-bubble bubble-assistant bubble-thinking">
              <span className="thinking-text">Thinking</span>
              <span className="thinking-dots">
                <span>.</span><span>.</span><span>.</span>
              </span>
            </div>
          </div>
        )}
      </div>

      <div className="assistant-suggestions-row">
        <span className="suggestions-label">Example questions:</span>
        <div className="suggestions-chips">
          {EXAMPLE_QUESTIONS.map((q, idx) => (
            <button
              key={idx}
              className="suggestion-chip-btn"
              onClick={() => handleSend(q)}
              disabled={loading}
            >
              • {q}
            </button>
          ))}
        </div>
      </div>

      <div className="assistant-input-bar">
        <input
          type="text"
          className="assistant-text-input"
          placeholder="Ask a question (e.g. 'Where should I park?')..."
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={handleKeyDown}
          disabled={loading}
        />
        <button
          className="assistant-send-btn"
          onClick={() => handleSend()}
          disabled={loading || !query.trim()}
        >
          {loading ? "Sending..." : "Send"}
        </button>
      </div>
    </div>
  )
}
