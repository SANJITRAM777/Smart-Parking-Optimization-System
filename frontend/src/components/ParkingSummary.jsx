import React from "react"

export default function ParkingSummary({ slots = [], loading = false }) {
  const totalSlots = slots.length || 4
  const availableSlots = slots.filter((s) => s.status?.toLowerCase() === "available").length
  const occupiedSlots = slots.filter((s) => s.status?.toLowerCase() === "occupied").length
  const pctFree = totalSlots > 0 ? Math.round((availableSlots / totalSlots) * 100) : 0

  return (
    <div className="driver-summary-card">
      <div className="summary-headline-wrap">
        <div className="summary-status-pill">
          <span className="live-pulse"></span>
          <span>LIVE FACILITY STATUS</span>
        </div>
        <h2 className="summary-main-count">
          {loading ? "Checking availability..." : `${availableSlots} of ${totalSlots} spaces available`}
        </h2>
        <p className="summary-subtext">
          {availableSlots === 0
            ? "Facility is currently at full capacity. Please wait for an open bay."
            : `${pctFree}% of spaces are currently ready for parking.`}
        </p>
      </div>

      <div className="summary-badges-row">
        <div className="summary-chip chip-available">
          <span className="chip-indicator"></span>
          <span className="chip-count">{availableSlots}</span>
          <span className="chip-label">AVAILABLE</span>
        </div>

        <div className="summary-chip chip-occupied">
          <span className="chip-indicator"></span>
          <span className="chip-count">{occupiedSlots}</span>
          <span className="chip-label">OCCUPIED</span>
        </div>
      </div>

      <div className="summary-progress-wrap">
        <div className="progress-bar-bg">
          <div
            className="progress-bar-fill"
            style={{ width: `${pctFree}%` }}
          ></div>
        </div>
      </div>
    </div>
  )
}
