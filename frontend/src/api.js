/**
 * API client module for communicating with FastAPI Backend
 */
const BASE_URL = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000"

export async function checkBackendHealth() {
  const res = await fetch(`${BASE_URL}/`, { method: "GET" })
  if (!res.ok) throw new Error(`Health check failed: ${res.statusText}`)
  return await res.json()
}

export async function fetchSlots() {
  const res = await fetch(`${BASE_URL}/api/slots`, { method: "GET" })
  if (!res.ok) throw new Error(`Failed to fetch slots: ${res.statusText}`)
  return await res.json()
}

export async function fetchRecommendation() {
  const res = await fetch(`${BASE_URL}/api/optimizer/recommendation`, { method: "GET" })
  if (!res.ok) throw new Error(`Failed to fetch recommendation: ${res.statusText}`)
  return await res.json()
}

export async function fetchPredictions() {
  const res = await fetch(`${BASE_URL}/api/dnn/predictions`, { method: "GET" })
  if (!res.ok) throw new Error(`Failed to fetch DNN predictions: ${res.statusText}`)
  return await res.json()
}

export async function fetchHistory(slotId = null, limit = 100) {
  let url = `${BASE_URL}/api/analytics/history?limit=${limit}`
  if (slotId && slotId !== "ALL") {
    url += `&slot_id=${encodeURIComponent(slotId)}`
  }
  const res = await fetch(url, { method: "GET" })
  if (!res.ok) throw new Error(`Failed to fetch historical data: ${res.statusText}`)
  return await res.json()
}

export async function sendGenAIChat(query) {
  const res = await fetch(`${BASE_URL}/api/genai/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query }),
  })
  if (!res.ok) throw new Error(`GenAI query failed: ${res.statusText}`)
  return await res.json()
}
