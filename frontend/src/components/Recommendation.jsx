import React, { useState } from "react"

export default function Recommendation({ recommendation = null, loading = false }) {
  const [navActive, setNavActive] = useState(false)

  if (loading) {
    return (
      <div className="recommendation-hero-card loading">
        <div className="hero-loading-text">
          <span className="spinner"></span> Finding your best parking space...
        </div>
      </div>
    )
  }

  const recSlot = recommendation?.recommended_slot
  const isFull = !recSlot || recommendation?.ranked_slots?.length === 0
  const dist = recommendation?.distance_m !== null && recommendation?.distance_m !== undefined
    ? Math.round(recommendation.distance_m)
    : null
  const predOcc = recommendation?.predicted_occupancy
  
  // Format driver-friendly occupancy risk
  let riskText = "Low chance of becoming occupied"
  if (predOcc !== null && predOcc !== undefined) {
    if (predOcc < 0.15) {
      riskText = "Very low chance of becoming occupied"
    } else if (predOcc < 0.40) {
      riskText = "Low chance of becoming occupied"
    } else if (predOcc < 0.65) {
      riskText = "Moderate chance of becoming occupied"
    } else {
      riskText = "Elevated chance of becoming occupied"
    }
  }

  if (isFull) {
    return (
      <div className="recommendation-hero-card hero-full">
        <div className="hero-full-content">
          <div className="hero-full-icon">⚠️</div>
          <div className="hero-full-text">
            <h3>No Parking Spaces Currently Available</h3>
            <p>
              All 4 parking bays are currently occupied. Please wait near the entrance;
              our real-time system will automatically recommend a bay as soon as a vehicle departs.
            </p>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="recommendation-hero-card">
      <div className="hero-header-badge">
        <span className="star-icon">⭐</span>
        <span className="badge-text">RECOMMENDED FOR YOU</span>
      </div>

      <div className="hero-main-display">
        <div className="hero-slot-id-large">{recSlot}</div>
        <div className="hero-key-info">
          <div className="hero-metric-item">
            <span className="hero-metric-icon">🚶</span>
            <span className="hero-metric-text">
              <strong>{dist !== null ? `${dist} m` : "Close"}</strong> walking distance
            </span>
          </div>

          <div className="hero-metric-item">
            <span className="hero-metric-icon">🛡️</span>
            <span className="hero-metric-text text-emerald">
              {riskText}
            </span>
          </div>

          <div className="hero-metric-item">
            <span className="hero-metric-icon">📍</span>
            <span className="hero-metric-text text-muted">
              {recSlot === "P1" ? "Near Entrance (Ground Floor)" :
               recSlot === "P2" ? "Ground Floor East" :
               recSlot === "P3" ? "Ground Floor West" : "Near Exit (Ground Floor)"}
            </span>
          </div>
        </div>
      </div>

      <div className="hero-action-row">
        <button
          className={`park-here-btn ${navActive ? "btn-active" : ""}`}
          onClick={() => setNavActive((prev) => !prev)}
        >
          {navActive ? "✓ NAVIGATION ACTIVE • HEAD TO " + recSlot : "PARK HERE →"}
        </button>

        {navActive && (
          <div className="nav-guidance-pill">
            Follow the green bay lights to <strong>Slot {recSlot}</strong> ({dist}m ahead)
          </div>
        )}
      </div>
    </div>
  )
}
