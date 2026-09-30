import React, { useState } from "react"
import DNNAnalytics from "./DNNAnalytics"

export default function SystemDetails({ recommendation = null, predictions = {}, refreshTrigger = 0 }) {
  const [isOpen, setIsOpen] = useState(false)
  const [activeTab, setActiveTab] = useState("overview")

  const rankedSlots = recommendation?.ranked_slots || []

  return (
    <div className="system-details-section">
      <button
        className="system-details-toggle-btn"
        onClick={() => setIsOpen((prev) => !prev)}
        aria-expanded={isOpen}
      >
        <span className="toggle-left">
          <span className="toggle-icon">⚙️</span>
          <span className="toggle-text">System Details & Technical Architecture</span>
          <span className="faculty-badge">Faculty / Viva View</span>
        </span>
        <span className="toggle-arrow">{isOpen ? "▲ Hide" : "▼ Expand"}</span>
      </button>

      {isOpen && (
        <div className="system-details-content">
          <div className="system-details-tabs">
            <button
              className={`details-tab-btn ${activeTab === "overview" ? "active" : ""}`}
              onClick={() => setActiveTab("overview")}
            >
              Architecture Overview
            </button>
            <button
              className={`details-tab-btn ${activeTab === "dnn" ? "active" : ""}`}
              onClick={() => setActiveTab("dnn")}
            >
              DNN & LSTM Telemetry
            </button>
            <button
              className={`details-tab-btn ${activeTab === "optimizer" ? "active" : ""}`}
              onClick={() => setActiveTab("optimizer")}
            >
              Optimizer Ranking
            </button>
          </div>

          {activeTab === "overview" && (
            <div className="tech-pillars-grid">
              <div className="tech-pillar-card">
                <div className="pillar-header">
                  <span className="pillar-icon">📡</span>
                  <h4>IoT Sensor</h4>
                </div>
                <p className="pillar-desc">
                  ESP32 and ultrasonic sensing provide parking occupancy data.
                </p>
                <div className="pillar-tag">Hardware: ESP32 + HC-SR04 (Bay P1)</div>
              </div>

              <div className="tech-pillar-card">
                <div className="pillar-header">
                  <span className="pillar-icon">🧠</span>
                  <h4>DNN Prediction</h4>
                </div>
                <p className="pillar-desc">
                  LSTM predicts future parking occupancy from recent parking history.
                </p>
                <div className="pillar-tag">TensorFlow / Keras Sequence Model</div>
              </div>

              <div className="tech-pillar-card">
                <div className="pillar-header">
                  <span className="pillar-icon">⚙️</span>
                  <h4>Optimization</h4>
                </div>
                <p className="pillar-desc">
                  The system selects a suitable available slot using availability, predicted occupancy and walking distance.
                </p>
                <div className="pillar-tag">Utility Formula: 0.5·Dist + 0.5·(1-Risk)</div>
              </div>

              <div className="tech-pillar-card">
                <div className="pillar-header">
                  <span className="pillar-icon">🤖</span>
                  <h4>GenAI</h4>
                </div>
                <p className="pillar-desc">
                  Gemini explains the parking recommendation and answers user questions using current system data.
                </p>
                <div className="pillar-tag">Google Gemini API (Live Context)</div>
              </div>
            </div>
          )}

          {activeTab === "dnn" && (
            <div className="tech-dnn-panel">
              <DNNAnalytics predictions={predictions} refreshTrigger={refreshTrigger} />
            </div>
          )}

          {activeTab === "optimizer" && (
            <div className="tech-optimizer-panel">
              <div className="panel-header">
                <h4>Optimizer Candidate Ranking</h4>
                <p className="panel-sub">
                  Calculated by deterministic utility scoring. Only available slots are eligible for recommendation.
                </p>
              </div>

              {rankedSlots.length === 0 ? (
                <div className="empty-ranking">All bays currently occupied. No available candidates.</div>
              ) : (
                <div className="ranking-table-responsive">
                  <table className="tech-ranking-table">
                    <thead>
                      <tr>
                        <th>Rank</th>
                        <th>Bay ID</th>
                        <th>Walking Distance</th>
                        <th>Predicted Occupancy</th>
                        <th>Utility Score</th>
                      </tr>
                    </thead>
                    <tbody>
                      {rankedSlots.map((slot, idx) => (
                        <tr key={slot.slot_id} className={idx === 0 ? "highlight-top-rank" : ""}>
                          <td>#{idx + 1}</td>
                          <td>
                            <strong>{slot.slot_id}</strong>
                            {idx === 0 && <span className="top-choice-tag">RECOMMENDED</span>}
                          </td>
                          <td>{Math.round(slot.distance_m)} meters</td>
                          <td>{(slot.predicted_occupancy * 100).toFixed(1)}%</td>
                          <td className="score-val">{slot.score.toFixed(4)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  )
}
