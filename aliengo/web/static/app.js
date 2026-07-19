"use strict";

const $ = (selector) => document.querySelector(selector);
const state = {
  clientId: null,
  snapshot: null,
  socket: null,
  reconnectTimer: null,
  currentCommandId: null,
  commandPoll: null,
  mediaRecorder: null,
  recordingTimer: null,
  recordingStarted: 0,
  chunks: [],
  confirmationCommandId: null,
  leaseTick: null,
  statePoll: null,
};

async function api(path, options = {}) {
  const headers = new Headers(options.headers || {});
  if (options.body && !(options.body instanceof FormData)) headers.set("Content-Type", "application/json");
  const response = await fetch(`/api/v1${path}`, { credentials: "same-origin", ...options, headers });
  if (response.status === 204) return null;
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    const error = new Error(payload.error?.message || `Request failed (${response.status})`);
    error.status = response.status;
    error.code = payload.error?.code;
    throw error;
  }
  return payload;
}

function showToast(message) {
  const toast = $("#toast");
  toast.textContent = message;
  toast.hidden = false;
  clearTimeout(showToast.timer);
  showToast.timer = setTimeout(() => { toast.hidden = true; }, 4500);
}

function setConnection(mode, label) {
  const pill = $("#connection-pill");
  pill.className = `connection-pill ${mode}`;
  pill.lastChild.textContent = ` ${label}`;
}

function showLogin() {
  $("#app").hidden = true;
  $("#login-screen").hidden = false;
  closeSocket();
}

function showApp() {
  $("#login-screen").hidden = true;
  $("#app").hidden = false;
}

async function refreshState() {
  try {
    const snapshot = await api("/state");
    state.snapshot = snapshot;
    renderState(snapshot);
    return snapshot;
  } catch (error) {
    if (error.status === 401) showLogin();
    throw error;
  }
}

function renderState(snapshot) {
  const robot = snapshot.robot;
  $("#backend-chip").textContent = snapshot.backend;
  $("#posture-value").textContent = robot.posture;
  $("#battery-value").textContent = `${Math.round(robot.battery_pct)}%`;
  $("#motion-value").textContent = robot.following ? "following" : robot.moving ? "moving" : "stationary";
  $("#heading-value").textContent = `${Math.round(robot.heading_deg)}°`;
  $("#position-value").textContent = `x ${robot.x.toFixed(2)} · y ${robot.y.toFixed(2)}`;

  const lease = snapshot.lease;
  const mine = Boolean(lease?.held_by_you);
  $("#lease-owner").textContent = lease ? lease.owner_name : "No active operator";
  $("#operator-avatar").textContent = lease ? lease.owner_name.slice(0, 2).toUpperCase() : "—";
  $("#lease-copy").textContent = mine ? "You have exclusive command control." : lease ? "State and e-stop remain available." : "Claim control to send commands.";
  $("#lease-time").textContent = lease ? lease.pinned ? "Pinned" : formatLeaseTime(lease.expires_in_s) : "Available";
  $("#claim-button").hidden = mine;
  $("#claim-button").disabled = Boolean(lease && !mine);
  $("#release-button").hidden = !mine;
  $("#reset-button").hidden = !mine;

  const canCommand = mine && !snapshot.busy && !snapshot.estop_active;
  $("#command-input").disabled = !canCommand;
  $("#send-button").disabled = !canCommand || !$("#command-input").value.trim();
  $("#voice-button").disabled = !canCommand || !navigator.mediaDevices?.getUserMedia || !window.MediaRecorder;
  $("#busy-chip").hidden = !snapshot.busy;

  const safetyBar = $("#safety-bar");
  safetyBar.classList.toggle("active", snapshot.estop_active);
  $("#safety-title").textContent = snapshot.estop_active ? "Emergency stop active" : "Emergency stop ready";
  $("#safety-copy").textContent = snapshot.estop_active ? "Motion is blocked until the control holder releases it." : "Available to every authenticated device.";
  $("#estop-button").disabled = snapshot.estop_active;
  $("#release-estop-button").hidden = !(snapshot.estop_active && mine && !snapshot.busy);
}

function formatLeaseTime(seconds) {
  const total = Math.max(0, Math.round(seconds));
  return `${Math.floor(total / 60)}:${String(total % 60).padStart(2, "0")}`;
}

function startLeaseTick() {
  clearInterval(state.leaseTick);
  state.leaseTick = setInterval(() => {
    const lease = state.snapshot?.lease;
    if (!lease || lease.pinned) return;
    lease.expires_in_s = Math.max(0, lease.expires_in_s - 1);
    $("#lease-time").textContent = formatLeaseTime(lease.expires_in_s);
    if (lease.expires_in_s <= 0) refreshState().catch(() => {});
  }, 1000);
}

function startStatePolling() {
  clearInterval(state.statePoll);
  state.statePoll = setInterval(() => refreshState().catch(() => {}), 5000);
}

function closeSocket() {
  clearTimeout(state.reconnectTimer);
  if (state.socket) {
    state.socket.onclose = null;
    state.socket.close();
    state.socket = null;
  }
}

function connectSocket() {
  closeSocket();
  const protocol = location.protocol === "https:" ? "wss" : "ws";
  const socket = new WebSocket(`${protocol}://${location.host}/api/v1/events`);
  state.socket = socket;
  setConnection("", "Connecting");
  socket.onopen = () => setConnection("online", "Live");
  socket.onmessage = ({ data }) => {
    const event = JSON.parse(data);
    handleEvent(event);
  };
  socket.onerror = () => setConnection("offline", "Offline");
  socket.onclose = () => {
    setConnection("offline", "Reconnecting");
    state.reconnectTimer = setTimeout(connectSocket, 1800);
  };
}

function handleEvent(event) {
  if (event.type === "snapshot") {
    state.snapshot = event.state;
    renderState(event.state);
    return;
  }
  if (event.type === "command_event") addCommandEvent(event);
  if (event.type === "confirmation_required") showConfirmation(event);
  if (event.type === "confirmation_resolved") closeConfirmation();
  if (event.type === "command_changed" && event.command_id === state.currentCommandId) {
    if (["completed", "failed", "cancelled"].includes(event.status)) loadCommand(event.command_id);
  }
  if (["state_changed", "lease_changed", "estop_changed", "command_changed"].includes(event.type)) {
    refreshState().catch(() => {});
  }
}

function addActivity(title, detail = "", tone = "", timestamp = new Date()) {
  $("#empty-activity")?.remove();
  const item = document.createElement("li");
  if (tone) item.classList.add(tone);
  const strong = document.createElement("strong");
  const span = document.createElement("span");
  const time = document.createElement("time");
  strong.textContent = title;
  span.textContent = detail;
  time.textContent = timestamp.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
  item.append(strong, span, time);
  $("#timeline").prepend(item);
}

function addCommandEvent(event) {
  const kind = event.kind;
  const payload = event.payload || {};
  if (kind === "tool_call") {
    const args = Object.entries(payload.args || {}).map(([key, value]) => `${key}=${value}`).join(", ");
    addActivity(`Tool · ${payload.skill}`, args || "No parameters");
  } else if (kind === "safety") {
    const tone = payload.decision === "block" ? "error" : payload.decision === "confirm" ? "warning" : "";
    addActivity(`Safety · ${payload.decision}`, payload.reason || payload.skill, tone);
  } else if (kind === "result") {
    addActivity(payload.success ? `Completed · ${payload.action}` : `Failed · ${payload.action}`, payload.error || "Robot action succeeded.", payload.success ? "" : "error");
  } else if (kind === "info") {
    addActivity("Agent notice", payload.text || "", "warning");
  }
}

async function loadCommand(commandId) {
  try {
    const command = await api(`/commands/${commandId}`);
    if (command.status === "completed") addActivity("AlienGo replied", command.reply || "Command completed.");
    else if (command.status === "failed") addActivity("Command failed", command.error || "Unknown failure", "error");
    if (["completed", "failed", "cancelled"].includes(command.status)) {
      clearInterval(state.commandPoll);
      state.commandPoll = null;
      state.currentCommandId = null;
      await refreshState();
    }
  } catch (error) {
    showToast(error.message);
  }
}

function pollCommand(commandId) {
  clearInterval(state.commandPoll);
  state.commandPoll = setInterval(() => loadCommand(commandId), 1500);
}

function showConfirmation(event) {
  state.confirmationCommandId = event.command_id;
  $("#confirmation-title").textContent = `Approve ${event.confirmation.skill}?`;
  $("#confirmation-reason").textContent = event.confirmation.reason;
  $("#confirmation-params").textContent = JSON.stringify(event.confirmation.params, null, 2);
  const bar = $("#confirmation-bar");
  bar.style.animation = "none";
  void bar.offsetWidth;
  bar.style.animation = `countdown ${event.timeout_s || 30}s linear forwards`;
  $("#confirmation-dialog").showModal();
}

function closeConfirmation() {
  const dialog = $("#confirmation-dialog");
  if (dialog.open) dialog.close();
  state.confirmationCommandId = null;
}

async function answerConfirmation(approved) {
  if (!state.confirmationCommandId) return;
  try {
    await api(`/commands/${state.confirmationCommandId}/confirmation`, {
      method: "POST", body: JSON.stringify({ approved }),
    });
    closeConfirmation();
  } catch (error) {
    showToast(error.message);
  }
}

async function startRecording() {
  if (state.mediaRecorder?.state === "recording") {
    state.mediaRecorder.stop();
    return;
  }
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const preferred = ["audio/webm;codecs=opus", "audio/ogg;codecs=opus", "audio/webm"]
      .find((type) => MediaRecorder.isTypeSupported(type));
    const recorder = preferred ? new MediaRecorder(stream, { mimeType: preferred }) : new MediaRecorder(stream);
    state.mediaRecorder = recorder;
    state.chunks = [];
    state.recordingStarted = Date.now();
    recorder.ondataavailable = (event) => { if (event.data.size) state.chunks.push(event.data); };
    recorder.onstop = async () => {
      clearTimeout(state.recordingTimer);
      stream.getTracks().forEach((track) => track.stop());
      $("#voice-button").classList.remove("recording");
      $("#voice-button").innerHTML = '<span aria-hidden="true">●</span> Record command';
      $("#voice-status").textContent = "Transcribing locally…";
      const type = recorder.mimeType || "audio/webm";
      const blob = new Blob(state.chunks, { type });
      const form = new FormData();
      form.append("audio", blob, type.includes("ogg") ? "command.ogg" : "command.webm");
      try {
        const result = await api("/transcriptions", { method: "POST", body: form });
        $("#command-input").value = result.text;
        $("#command-input").focus();
        $("#send-button").disabled = !result.text.trim();
        $("#voice-status").textContent = `${result.duration_s}s transcribed · review before sending`;
        addActivity("Voice transcribed", "Review the text before sending.");
      } catch (error) {
        $("#voice-status").textContent = "Transcription failed · try again";
        showToast(error.message);
      }
    };
    recorder.start(250);
    $("#voice-button").classList.add("recording");
    $("#voice-button").textContent = "Stop recording";
    $("#voice-status").textContent = "Listening… tap stop when finished";
    state.recordingTimer = setTimeout(() => {
      if (recorder.state === "recording") recorder.stop();
    }, 15000);
  } catch (error) {
    showToast(`Microphone unavailable: ${error.message}`);
  }
}

$("#login-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const button = event.submitter;
  button.disabled = true;
  try {
    const result = await api("/sessions", {
      method: "POST",
      body: JSON.stringify({ display_name: $("#display-name").value, passcode: $("#passcode").value }),
    });
    state.clientId = result.client_id;
    localStorage.setItem("aliengo_operator_name", result.display_name);
    $("#passcode").value = "";
    showApp();
    await refreshState();
    connectSocket();
    startLeaseTick();
    startStatePolling();
  } catch (error) {
    showToast(error.message);
  } finally {
    button.disabled = false;
  }
});

$("#claim-button").addEventListener("click", async () => {
  try { await api("/lease", { method: "PUT" }); await refreshState(); }
  catch (error) { showToast(error.message); }
});

$("#release-button").addEventListener("click", async () => {
  try { await api("/lease", { method: "DELETE" }); await refreshState(); }
  catch (error) { showToast(error.message); }
});

$("#estop-button").addEventListener("click", async () => {
  try { await api("/estop", { method: "POST" }); await refreshState(); addActivity("Emergency stop activated", "All motion has been halted.", "error"); }
  catch (error) { showToast(error.message); }
});

$("#release-estop-button").addEventListener("click", async () => {
  if (!window.confirm("Release the emergency stop? This only re-enables commands; it does not start motion.")) return;
  try { await api("/estop", { method: "DELETE" }); await refreshState(); }
  catch (error) { showToast(error.message); }
});

$("#reset-button").addEventListener("click", async () => {
  if (!window.confirm("Reset robot state and every client's conversation history?")) return;
  try { await api("/reset", { method: "POST" }); await refreshState(); addActivity("System reset", "Robot state and conversation histories were cleared."); }
  catch (error) { showToast(error.message); }
});

$("#send-button").addEventListener("click", async () => {
  const text = $("#command-input").value.trim();
  if (!text) return;
  try {
    const job = await api("/commands", { method: "POST", body: JSON.stringify({ text }) });
    state.currentCommandId = job.id;
    $("#command-input").value = "";
    addActivity("Command submitted", text);
    pollCommand(job.id);
    await refreshState();
  } catch (error) { showToast(error.message); }
});

$("#command-input").addEventListener("input", () => {
  const canSend = state.snapshot?.lease?.held_by_you && !state.snapshot?.busy && !state.snapshot?.estop_active;
  $("#send-button").disabled = !canSend || !$("#command-input").value.trim();
});
$("#command-input").addEventListener("keydown", (event) => {
  if ((event.ctrlKey || event.metaKey) && event.key === "Enter" && !$("#send-button").disabled) $("#send-button").click();
});
$("#voice-button").addEventListener("click", startRecording);
$("#approve-button").addEventListener("click", () => answerConfirmation(true));
$("#decline-button").addEventListener("click", () => answerConfirmation(false));
$("#clear-activity").addEventListener("click", () => {
  $("#timeline").replaceChildren();
  const empty = document.createElement("li");
  empty.className = "empty-state";
  empty.id = "empty-activity";
  const strong = document.createElement("strong");
  const span = document.createElement("span");
  strong.textContent = "Waiting for a command";
  span.textContent = "Tool calls and safety decisions will appear here.";
  empty.append(strong, span);
  $("#timeline").append(empty);
});
$("#logout-button").addEventListener("click", async () => {
  try { await api("/sessions/current", { method: "DELETE" }); }
  catch (_) { /* Local logout still clears the UI. */ }
  showLogin();
});

async function bootstrap() {
  $("#display-name").value = localStorage.getItem("aliengo_operator_name") || "";
  try {
    await refreshState();
    showApp();
    connectSocket();
    startLeaseTick();
    startStatePolling();
  } catch (_) {
    showLogin();
  }
}

bootstrap();
