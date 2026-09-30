import React from "react"

const DEFAULT_DISTANCES = {
  P1: 10.0,
  P2: 25.0,
  P3: 40.0,
  P4: 15.0,
}

export default function ParkingSlot({ slot, prediction, isRecommended = false }) {
  const slotId = slot.slot_id
  const isAvailable = slot.status?.toLowerCase() === "available"
  const distance = Math.round(slot.distance_m || DEFAULT_DISTANCES[slotId] || 15.0)
  const prob = prediction?.probability !== undefined ? prediction.probability : 0.05

  // Driver-friendly chance label
  let chanceLabel = "Low chance of becoming occupied"
  if (prob < 0.20) {
    chanceLabel = "Low chance of filling"
  } else if (prob < 0.50) {
    chanceLabel = "Moderate chance of filling"
  } else {
    chanceLabel = "Likely to fill soon"
  }

  return (
    <div
      className={`driver-slot-card ${isAvailable ? "card-free" : "card-occupied"} ${
        isRecommended ? "card-recommended" : ""
      }`}
    >
      {isRecommended && (
        <div className="card-recommended-banner">
          <span>⭐ RECOMMENDED</span>
        </div>
      )}

      <div className="slot-card-header">
        <div className="slot-title-group">
          <span className="slot-id-text">{slotId}</span>
          {slotId === "P1" && (
            <span className="sensor-source-tag">IoT Sensor</span>
          )}
        </div>

        <div className={`status-badge ${isAvailable ? "badge-free" : "badge-occupied"}`}>
          <span className="status-dot"></span>
          <span>{isAvailable ? "FREE" : "OCCUPIED"}</span>
        </div>
      </div>

      <div className="slot-visual-bay">
        <div className={`bay-car-silhouette ${isAvailable ? "silhouette-free" : "silhouette-occupied"}`}>
          <span className="bay-emoji">{isAvailable ? "🅿️" : "🚘"}</span>
        </div>
      </div>

      <div className="slot-meta-group">
        {isAvailable ? (
          <>
            <div className="meta-line">
              <span className="meta-icon">🚶</span>
              <span className="meta-val">{distance} m walk</span>
            </div>
            <div className="meta-sub-line">
              <span className="sub-tag">{chanceLabel}</span>
            </div>
          </>
        ) : (
          <>
            <div className="meta-line text-rose">
              <span className="meta-icon">🔒</span>
              <span className="meta-val">Currently occupied</span>
            </div>
            <div className="meta-sub-line">
              <span className="sub-tag text-muted">Vehicle parked</span>
            </div>
          </>
        )}
      </div>
    </div>
  )
}
