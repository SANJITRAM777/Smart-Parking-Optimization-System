import React, { useState, useEffect, useCallback } from "react"
import { fetchSlots, fetchRecommendation, fetchPredictions, checkBackendHealth } from "./api"
import ParkingSummary from "./components/ParkingSummary"
import ParkingSlot from "./components/ParkingSlot"
import Recommendation from "./components/Recommendation"
import GenAIAssistant from "./components/GenAIAssistant"
import SystemDetails from "./components/SystemDetails"
import "./App.css"

export default function App() {
  const [slots, setSlots] = useState([])
  const [recommendation, setRecommendation] = useState(null)
  const [predictions, setPredictions] = useState({})
  const [loading, setLoading] = useState(true)
  const [isRefreshing, setIsRefreshing] = useState(false)
  const [backendOnline, setBackendOnline] = useState(true)
  const [errorMessage, setErrorMessage] = useState(null)
  const [autoRefresh, setAutoRefresh] = useState(true)
  const [lastRefreshedAt, setLastRefreshedAt] = useState(new Date())
  const [refreshTrigger, setRefreshTrigger] = useState(0)

  const loadData = useCallback(async (isManual = false) => {
    if (isManual) setIsRefreshing(true)
    setErrorMessage(null)

    try {
      const [slotsData, recData, predData] = await Promise.all([
        fetchSlots(),
        fetchRecommendation(),
        fetchPredictions(),
      ])

      setSlots(slotsData || [])
      setRecommendation(recData || null)
      setPredictions(predData || {})
      setBackendOnline(true)
      setLastRefreshedAt(new Date())
      setRefreshTrigger((c) => c + 1)
    } catch (err) {
      console.error("Data fetch failed:", err)
      setBackendOnline(false)
      setErrorMessage("Backend unavailable. Please ensure FastAPI server is running on port 8000.")
    } finally {
      setLoading(false)
      if (isManual) setIsRefreshing(false)
    }
  }, [])

  useEffect(() => {
    loadData(false)
  }, [loadData])

  // Periodic live refresh (defaults to active 4s polling for live ESP32 updates)
  useEffect(() => {
    if (!autoRefresh) return
    const interval = setInterval(() => {
      loadData(false)
    }, 4000)
    return () => clearInterval(interval)
  }, [autoRefresh, loadData])

  const recSlotId = recommendation?.recommended_slot

  return (
    <div className="driver-app-shell">
      {/* Driver-Centric Header */}
      <header className="driver-header">
        <div className="header-inner">
          <div className="brand-block">
            <div className="logo-row">
              <span className="brand-icon">🅿️</span>
              <h1 className="brand-title">SMART PARKING</h1>
            </div>
            <p className="brand-tagline">Find your best parking space</p>
          </div>

          <div className="header-controls">
            <div className={`connection-badge ${backendOnline ? "online" : "offline"}`}>
              <span className="status-orb"></span>
              <span>{backendOnline ? "Live Updates" : "Backend Offline"}</span>
            </div>

            <div className="polling-toggle">
              <label className="toggle-switch">
                <input
                  type="checkbox"
                  checked={autoRefresh}
                  onChange={(e) => setAutoRefresh(e.target.checked)}
                />
                <span className="toggle-slider"></span>
              </label>
              <span className="toggle-caption">Auto-Sync</span>
            </div>

            <button
              className={`driver-refresh-btn ${isRefreshing ? "refreshing" : ""}`}
              onClick={() => loadData(true)}
              disabled={isRefreshing}
              title="Refresh parking telemetry"
            >
              <span className="refresh-symbol">↻</span>
              <span>{isRefreshing ? "Updating..." : "Refresh"}</span>
            </button>
          </div>
        </div>
      </header>

      {/* Backend Offline Warning Banner */}
      {!backendOnline && (
        <div className="driver-alert-strip">
          <span className="alert-strip-icon">⚠️</span>
          <div className="alert-strip-text">
            <strong>System Connection Notice:</strong> {errorMessage}
          </div>
        </div>
      )}

      {/* Main Driver Dashboard */}
      <main className="driver-main-container">
        {/* Top Summary & Capacity */}
        <section className="dashboard-section summary-section">
          <ParkingSummary slots={slots} loading={loading} />
        </section>

        {/* Hero Section: Recommended Parking Space */}
        <section className="dashboard-section recommendation-section">
          <Recommendation recommendation={recommendation} loading={loading} />
        </section>

        {/* Parking Spaces Grid */}
        <section className="dashboard-section bays-section">
          <div className="section-header-row">
            <h2 className="section-heading">AVAILABLE PARKING</h2>
            <span className="sync-timestamp">
              Updated {lastRefreshedAt.toLocaleTimeString()}
            </span>
          </div>

          <div className="driver-bays-grid">
            {["P1", "P2", "P3", "P4"].map((slotId) => {
              const slotObj = slots.find((s) => s.slot_id === slotId) || {
                slot_id: slotId,
                status: "available",
                location_label: `Bay ${slotId}`,
              }
              const predObj = predictions[slotId]
              const isRec = recSlotId === slotId

              return (
                <ParkingSlot
                  key={slotId}
                  slot={slotObj}
                  prediction={predObj}
                  isRecommended={isRec}
                />
              )
            })}
          </div>
        </section>

        {/* Live GenAI Parking Assistant */}
        <section className="dashboard-section assistant-section">
          <GenAIAssistant />
        </section>

        {/* Expandable Faculty / Technical Inspection Section */}
        <section className="dashboard-section faculty-section">
          <SystemDetails
            recommendation={recommendation}
            predictions={predictions}
            refreshTrigger={refreshTrigger}
          />
        </section>
      </main>

      {/* Driver Footer */}
      <footer className="driver-footer">
        <div className="footer-inner">
          <p className="footer-primary">
            <strong>Smart Parking Optimization System</strong> • IoT + DNN + Optimization + GenAI
          </p>
          <p className="footer-secondary">
            ESP32 Ultrasonic Telemetry • LSTM Occupancy Prediction • Utility Optimizer • Live Gemini Assistant
          </p>
        </div>
      </footer>
    </div>
  )
}
