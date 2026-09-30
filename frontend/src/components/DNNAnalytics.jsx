import React, { useState, useEffect } from "react"
import { fetchHistory } from "../api"

export default function DNNAnalytics({ predictions = {}, refreshTrigger = 0 }) {
  const [selectedSlot, setSelectedSlot] = useState("ALL")
  const [historyData, setHistoryData] = useState([])
  const [loadingHistory, setLoadingHistory] = useState(false)
  const [historyError, setHistoryError] = useState(null)

  useEffect(() => {
    let isMounted = true
    setLoadingHistory(true)
    setHistoryError(null)

    fetchHistory(selectedSlot, 50)
      .then((data) => {
        if (isMounted) {
          setHistoryData(data || [])
          setLoadingHistory(false)
        }
      })
      .catch((err) => {
        if (isMounted) {
          setHistoryError(err.message)
          setLoadingHistory(false)
        }
      })

    return () => {
      isMounted = false
    }
  }, [selectedSlot, refreshTrigger])

  const slotKeys = ["P1", "P2", "P3", "P4"]

  const reversedLogs = [...historyData].reverse()
  const chartPoints = reversedLogs.map((item, idx) => ({
    x: idx,
    time: item.timestamp ? item.timestamp.split(" ")[1] || item.timestamp : String(idx),
    occupied: item.occupied,
    distance: item.distance_cm,
    slot_id: item.slot_id,
  }))

  const svgWidth = 600
  const svgHeight = 180
  const padding = { top: 20, right: 20, bottom: 30, left: 60 }
  const plotWidth = svgWidth - padding.left - padding.right
  const plotHeight = svgHeight - padding.top - padding.bottom

  const getPointsString = () => {
    if (chartPoints.length < 2) return ""
    return chartPoints
      .map((p, i) => {
        const px = padding.left + (i / (chartPoints.length - 1)) * plotWidth
        const py = padding.top + (1 - p.occupied) * plotHeight
        return `${px},${py}`
      })
      .join(" ")
  }

  return (
    <div className="section-card dnn-analytics-card">
      <div className="section-title-wrap">
        <div className="section-title">
          <span className="icon">🧠</span>
          <h2>DNN Analytics & LSTM Forecasting</h2>
        </div>
        <div className="badge-source">TensorFlow / Keras LSTM Model (Window: 8)</div>
      </div>

      <div className="dnn-grid-layout">
        <div className="dnn-predictions-col">
          <div className="sub-section-header">
            <h3>LSTM Predicted Occupancy Probability</h3>
            <span className="info-chip">Next-Interval Forecast</span>
          </div>

          <div className="predictions-cards-grid">
            {slotKeys.map((sId) => {
              const predInfo = predictions[sId] || { probability: 0.05, prediction: "AVAILABLE", sequence_used: [0, 0, 0, 0, 0, 0, 0, 0] }
              const prob = predInfo.probability !== undefined ? predInfo.probability : 0.0
              const probPct = (prob * 100).toFixed(1)
              const isHighRisk = prob >= 0.5
              const seq = predInfo.sequence_used || []

              return (
                <div key={sId} className={`pred-mini-card ${isHighRisk ? "pred-card-high" : "pred-card-low"}`}>
                  <div className="pred-card-header">
                    <span className="pred-slot-id">{sId}</span>
                    <span className={`pred-status-tag ${isHighRisk ? "tag-occupied" : "tag-available"}`}>
                      {predInfo.prediction || (isHighRisk ? "OCCUPIED" : "AVAILABLE")}
                    </span>
                  </div>

                  <div className="pred-metric-box">
                    <div className="pred-pct-val">{probPct}%</div>
                    <div className="pred-pct-label">Predicted Risk</div>
                  </div>

                  <div className="pred-seq-row">
                    <span className="pred-seq-label">History Window (8):</span>
                    <div className="seq-dots">
                      {seq.map((val, sIdx) => (
                        <span
                          key={sIdx}
                          className={`seq-dot ${val === 1 ? "seq-occ" : "seq-avail"}`}
                          title={`T-${8 - sIdx}: ${val === 1 ? "Occupied" : "Free"}`}
                        ></span>
                      ))}
                    </div>
                  </div>
                </div>
              )
            })}
          </div>

          <div className="dnn-disclaimer-box">
            <span className="disclaimer-icon">ℹ️</span>
            <div className="disclaimer-text">
              <strong>Evaluation Note:</strong> The LSTM neural network was trained on sequential time-series patterns.
              The reported training accuracy (96.48%) represents historical benchmark convergence and does not guarantee future state certainty in live random environments.
            </div>
          </div>
        </div>

        <div className="dnn-history-col">
          <div className="sub-section-header history-header">
            <div>
              <h3>Time vs Parking Occupancy</h3>
              <span className="info-chip">SQLite Telemetry Stream</span>
            </div>

            <div className="slot-filter-pills">
              {["ALL", "P1", "P2", "P3", "P4"].map((filterOpt) => (
                <button
                  key={filterOpt}
                  className={`filter-pill ${selectedSlot === filterOpt ? "active" : ""}`}
                  onClick={() => setSelectedSlot(filterOpt)}
                >
                  {filterOpt}
                </button>
              ))}
            </div>
          </div>

          <div className="chart-container">
            {loadingHistory ? (
              <div className="chart-loading">Loading telemetry history logs...</div>
            ) : historyError ? (
              <div className="chart-error">Unable to load telemetry: {historyError}</div>
            ) : chartPoints.length === 0 ? (
              <div className="chart-empty">No sensor logs found in SQLite database yet.</div>
            ) : (
              <div className="svg-chart-wrapper">
                <svg viewBox={`0 0 ${svgWidth} ${svgHeight}`} className="analytics-svg" preserveAspectRatio="none">
                  <line x1={padding.left} y1={padding.top} x2={svgWidth - padding.right} y2={padding.top} stroke="rgba(255,255,255,0.08)" strokeDasharray="3 3" />
                  <line x1={padding.left} y1={padding.top + plotHeight / 2} x2={svgWidth - padding.right} y2={padding.top + plotHeight / 2} stroke="rgba(255,255,255,0.08)" strokeDasharray="3 3" />
                  <line x1={padding.left} y1={padding.top + plotHeight} x2={svgWidth - padding.right} y2={padding.top + plotHeight} stroke="rgba(255,255,255,0.15)" />

                  <text x={padding.left - 8} y={padding.top + 4} fill="#ef4444" fontSize="10" textAnchor="end" fontWeight="600">OCCUPIED</text>
                  <text x={padding.left - 8} y={padding.top + plotHeight + 4} fill="#10b981" fontSize="10" textAnchor="end" fontWeight="600">AVAILABLE</text>

                  {chartPoints.length > 1 ? (
                    <polygon
                      points={`${padding.left},${padding.top + plotHeight} ${getPointsString()} ${padding.left + plotWidth},${padding.top + plotHeight}`}
                      fill="url(#chartGradient)"
                      opacity="0.25"
                    />
                  ) : null}

                  {chartPoints.length > 1 ? (
                    <polyline
                      fill="none"
                      stroke="#06b6d4"
                      strokeWidth="2.5"
                      points={getPointsString()}
                    />
                  ) : null}

                  {chartPoints.map((p, i) => {
                    const px = padding.left + (chartPoints.length === 1 ? plotWidth / 2 : (i / (chartPoints.length - 1)) * plotWidth)
                    const py = padding.top + (1 - p.occupied) * plotHeight
                    return (
                      <g key={i} className="chart-dot-group">
                        <circle
                          cx={px}
                          cy={py}
                          r="4"
                          fill={p.occupied === 1 ? "#ef4444" : "#10b981"}
                          stroke="#0b0f19"
                          strokeWidth="2"
                        />
                      </g>
                    )
                  })}

                  <defs>
                    <linearGradient id="chartGradient" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="#06b6d4" stopOpacity="0.8" />
                      <stop offset="100%" stopColor="#06b6d4" stopOpacity="0.0" />
                    </linearGradient>
                  </defs>
                </svg>
                <div className="chart-axis-footer">
                  <span>← Older Telemetry ({chartPoints.length} records)</span>
                  <span>Latest Sensor Readings →</span>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
