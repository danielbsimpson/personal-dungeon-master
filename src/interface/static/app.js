/**
 * Personal Dungeon Master — Web Frontend (Phase 12)
 *
 * Handles:
 *  - Setup screen: campaign + personality selection → POST /session/start
 *  - WebSocket chat loop: token streaming, dice panels, interrupts
 *  - REST commands: /status, /journal, /graph, /roll, /rules
 *  - Three.js avatar: procedural blocky figure per DM personality
 *  - Scene image fade-in (Phase 13 hook — wired once backend sends it)
 */

'use strict';

/* ═══════════════════════════════════════════════════════════════
   STATE
═══════════════════════════════════════════════════════════════ */
const state = {
  sessionId: null,
  ws: null,
  streaming: false,
  currentDmBubble: null,   // the <div> receiving streaming tokens
  personalities: [],
  selectedPersonality: 'The Sage',
  avatar: null,            // DmAvatar instance
};

/* ═══════════════════════════════════════════════════════════════
   DOM REFS
═══════════════════════════════════════════════════════════════ */
const $ = id => document.getElementById(id);
const setupScreen       = $('setup-screen');
const gameScreen        = $('game-screen');
const campaignSelect    = $('campaign-select');
const personalityGrid   = $('personality-grid');
const startBtn          = $('start-btn');
const startBtnText      = $('start-btn-text');
const startSpinner      = $('start-spinner');
const setupError        = $('setup-error');
const chatLog           = $('chat-log');
const playerInput       = $('player-input');
const sendBtn           = $('send-btn');
const interruptBtn      = $('interrupt-btn');
const personalityLabel  = $('personality-label');
const switchBtn         = $('switch-personality-btn');
const sceneImage        = $('scene-image');
const imageCaption      = $('image-caption');
const statusPanel       = $('status-panel');
const personalityModal  = $('personality-modal');
const modalGrid         = $('modal-personality-grid');
const modalCancel       = $('modal-cancel');
const inputModal        = $('input-modal');
const inputModalTitle   = $('input-modal-title');
const inputModalField   = $('input-modal-field');
const inputModalCancel  = $('input-modal-cancel');
const inputModalConfirm = $('input-modal-confirm');

/* ═══════════════════════════════════════════════════════════════
   INIT — fetch metadata and populate setup screen
═══════════════════════════════════════════════════════════════ */
async function init() {
  const [campaigns, personalities] = await Promise.all([
    fetch('/campaigns').then(r => r.json()),
    fetch('/personalities').then(r => r.json()),
  ]);

  state.personalities = personalities;

  // Populate campaign dropdown
  campaignSelect.innerHTML = campaigns.map(c =>
    `<option value="${c.name}">${c.name}</option>`
  ).join('');
  if (campaigns.length > 0) startBtn.disabled = false;

  // Build personality cards
  buildPersonalityGrid(personalityGrid, personalities, (p) => {
    state.selectedPersonality = p.name;
    document.querySelectorAll('#personality-grid .p-card').forEach(el => {
      el.classList.toggle('selected', el.dataset.name === p.name);
    });
  });

  // Select The Sage by default
  const defaultCard = personalityGrid.querySelector('[data-name="The Sage"]');
  if (defaultCard) defaultCard.classList.add('selected');
}

function buildPersonalityGrid(container, personalities, onSelect) {
  container.innerHTML = personalities.map(p => `
    <div class="p-card" data-name="${p.name}" title="${p.description}">
      <div class="p-name">${p.name}</div>
      <div class="p-meta">${p.tone} · ${p.verbosity}</div>
    </div>
  `).join('');
  container.querySelectorAll('.p-card').forEach(card => {
    card.addEventListener('click', () => {
      const p = personalities.find(p => p.name === card.dataset.name);
      if (p) onSelect(p);
    });
  });
}

/* ═══════════════════════════════════════════════════════════════
   START SESSION
═══════════════════════════════════════════════════════════════ */
startBtn.addEventListener('click', async () => {
  const campaign = campaignSelect.value;
  if (!campaign) return;

  startBtn.disabled = true;
  startBtnText.textContent = 'Loading…';
  startSpinner.classList.remove('hidden');
  setupError.classList.add('hidden');

  try {
    const res = await fetch('/session/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ campaign, personality: state.selectedPersonality }),
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'Server error');
    }
    const data = await res.json();
    state.sessionId = data.session_id;

    // Show game screen
    setupScreen.classList.add('hidden');
    gameScreen.classList.remove('hidden');

    // Boot avatar with selected personality
    const pData = state.personalities.find(p => p.name === data.personality);
    personalityLabel.textContent = data.personality;
    state.avatar = new DmAvatar($('dm-canvas'), pData || state.personalities[0]);

    // Open WebSocket
    connectWs();
  } catch (err) {
    setupError.textContent = `Error: ${err.message}`;
    setupError.classList.remove('hidden');
    startBtn.disabled = false;
    startBtnText.textContent = 'Begin Adventure';
    startSpinner.classList.add('hidden');
  }
});

/* ═══════════════════════════════════════════════════════════════
   WEBSOCKET
═══════════════════════════════════════════════════════════════ */
function connectWs() {
  const wsUrl = `ws://${location.host}/session/${state.sessionId}/chat`;
  state.ws = new WebSocket(wsUrl);

  state.ws.addEventListener('message', evt => {
    const msg = JSON.parse(evt.data);
    handleServerMsg(msg);
  });

  state.ws.addEventListener('close', () => {
    if (state.streaming) finishStreaming();
    addSystemMsg('Connection closed.');
  });

  state.ws.addEventListener('error', () => {
    addSystemMsg('WebSocket error — please reload.');
  });
}

function sendWs(payload) {
  if (state.ws && state.ws.readyState === WebSocket.OPEN) {
    state.ws.send(JSON.stringify(payload));
  }
}

/* ═══════════════════════════════════════════════════════════════
   SERVER MESSAGE DISPATCH
═══════════════════════════════════════════════════════════════ */
function handleServerMsg(msg) {
  switch (msg.type) {
    case 'opening':
      appendDmMessage(msg.content, false);
      break;

    case 'token':
      if (!state.streaming) beginStreaming();
      appendToken(msg.content);
      break;

    case 'roll':
      appendRollCard(msg.result);
      break;

    case 'interrupted':
      if (state.currentDmBubble) {
        state.currentDmBubble.classList.remove('streaming');
        state.currentDmBubble.classList.add('interrupted');
      }
      break;

    case 'done':
      finishStreaming();
      break;

    case 'reconnected':
      addSystemMsg('Reconnected to session.');
      break;

    case 'error':
      addSystemMsg(`⚠ ${msg.message}`);
      finishStreaming();
      break;

    case 'scene_image':
      showSceneImage(msg.url, msg.caption || '');
      break;
  }
}

/* ═══════════════════════════════════════════════════════════════
   STREAMING HELPERS
═══════════════════════════════════════════════════════════════ */
function beginStreaming() {
  state.streaming = true;
  setInputEnabled(false);
  interruptBtn.classList.remove('hidden');
  sendBtn.classList.add('hidden');

  // Create a new DM bubble that will receive tokens
  const wrap = document.createElement('div');
  wrap.innerHTML = '<div class="msg-label">Dungeon Master</div>';
  const bubble = document.createElement('div');
  bubble.className = 'msg-dm streaming';
  wrap.appendChild(bubble);
  chatLog.appendChild(wrap);
  state.currentDmBubble = bubble;
  state.currentDmBubble._rawText = '';
  scrollToBottom();
}

function appendToken(token) {
  if (!state.currentDmBubble) beginStreaming();
  state.currentDmBubble._rawText += token;
  state.currentDmBubble.innerHTML = renderMarkdown(state.currentDmBubble._rawText);
  scrollToBottom();
}

function finishStreaming() {
  state.streaming = false;
  if (state.currentDmBubble) {
    state.currentDmBubble.classList.remove('streaming');
    state.currentDmBubble = null;
  }
  setInputEnabled(true);
  interruptBtn.classList.add('hidden');
  sendBtn.classList.remove('hidden');
}

/* ═══════════════════════════════════════════════════════════════
   CHAT RENDERING
═══════════════════════════════════════════════════════════════ */
function appendDmMessage(text, opening = false) {
  const wrap = document.createElement('div');
  if (opening) wrap.innerHTML = '<div class="msg-label">Dungeon Master</div>';
  const bubble = document.createElement('div');
  bubble.className = 'msg-dm';
  bubble.innerHTML = renderMarkdown(text);
  wrap.appendChild(bubble);
  chatLog.appendChild(wrap);
  scrollToBottom();
}

function appendPlayerMessage(text) {
  const wrap = document.createElement('div');
  wrap.innerHTML = `<div class="msg-label" style="text-align:right">You</div>`;
  const bubble = document.createElement('div');
  bubble.className = 'msg-player';
  bubble.textContent = text;
  wrap.appendChild(bubble);
  chatLog.appendChild(wrap);
  scrollToBottom();
}

function appendRollCard(result) {
  const modStr = result.modifier > 0 ? `+${result.modifier}` :
                 result.modifier < 0 ? `${result.modifier}` : '';
  const rollsStr = result.rolls.join(', ');
  const div = document.createElement('div');
  div.className = 'roll-card';
  div.innerHTML = `
    <span class="roll-badge">d${result.die}</span>
    <div>
      <div class="roll-label">🎲 ${result.label}</div>
      <div class="roll-detail">Rolls: ${rollsStr}${modStr ? '  ' + modStr : ''}</div>
    </div>
    <div class="roll-total">${result.total}</div>
  `;
  chatLog.appendChild(div);
  scrollToBottom();
}

function addSystemMsg(text) {
  const div = document.createElement('div');
  div.className = 'msg-system';
  div.textContent = text;
  chatLog.appendChild(div);
  scrollToBottom();
}

function scrollToBottom() {
  chatLog.scrollTop = chatLog.scrollHeight;
}

/* ═══════════════════════════════════════════════════════════════
   PLAYER INPUT
═══════════════════════════════════════════════════════════════ */
function sendMessage() {
  const text = playerInput.value.trim();
  if (!text || state.streaming) return;
  appendPlayerMessage(text);
  sendWs({ type: 'message', content: text });
  playerInput.value = '';
}

sendBtn.addEventListener('click', sendMessage);
playerInput.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
});
interruptBtn.addEventListener('click', () => {
  sendWs({ type: 'interrupt' });
});

function setInputEnabled(enabled) {
  playerInput.disabled = !enabled;
  sendBtn.disabled = !enabled;
  if (enabled) playerInput.focus();
}

/* ═══════════════════════════════════════════════════════════════
   QUICK COMMANDS
═══════════════════════════════════════════════════════════════ */
document.querySelectorAll('.quick-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    const cmd = btn.dataset.cmd;
    if (cmd === 'roll')  return promptModal('Roll Dice', 'e.g. d20+3 or 2d6', handleRoll);
    if (cmd === 'rules') return promptModal('Rules Lookup', 'e.g. grapple, concentration', handleRules);
    if (cmd === 'status')  runCommand({ command: 'status' });
    if (cmd === 'journal') runCommand({ command: 'journal' });
  });
});

async function runCommand(body) {
  try {
    const res = await fetch(`/session/${state.sessionId}/command`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    const data = await res.json();
    if (!res.ok) { addSystemMsg(`⚠ ${data.detail}`); return; }
    renderCommandResult(data);
  } catch (err) {
    addSystemMsg(`⚠ Command error: ${err.message}`);
  }
}

function handleRoll(expr) { runCommand({ command: 'roll', expr }); }
function handleRules(topic) { runCommand({ command: 'rules', topic }); }

function renderCommandResult(data) {
  if (data.command === 'status') {
    appendStatusCard(data.data);
  } else if (data.command === 'journal') {
    appendJournalCard(data.data);
  } else if (data.command === 'graph') {
    appendGraphCard(data.entity, data.data);
  } else if (data.command === 'roll') {
    appendRollCard(data.data);
  } else if (data.command === 'rules') {
    appendInfoCard(`Rules: ${data.topic}`, renderMarkdown(data.data));
  } else if (data.command === 'reset') {
    addSystemMsg('Session window cleared.');
  } else if (data.command === 'fullreset') {
    addSystemMsg('Full reset complete. Knowledge graph wiped.');
  } else if (data.command === 'personality') {
    const p = data.data;
    personalityLabel.textContent = p.name;
    addSystemMsg(`DM personality switched to ${p.name}.`);
    const pData = state.personalities.find(x => x.name === p.name);
    if (pData && state.avatar) state.avatar.switchPersonality(pData);
  }
}

function appendStatusCard(char) {
  const s = char.ability_scores;
  const div = document.createElement('div');
  div.className = 'info-card';
  div.innerHTML = `
    <h3>⚔ ${char.name}</h3>
    <table>
      <tr><th>Race</th><td class="stat-val">${char.race}</td>
          <th>Class</th><td class="stat-val">${char.character_class}${char.subclass ? ' (' + char.subclass + ')' : ''} ${char.level}</td></tr>
      <tr><th>HP</th><td class="stat-val">${char.hit_points}</td>
          <th>AC</th><td class="stat-val">${char.armor_class}</td></tr>
      <tr><th>Speed</th><td class="stat-val">${char.speed} ft</td>
          <th>Prof</th><td class="stat-val">+${char.proficiency_bonus}</td></tr>
    </table>
    <div style="margin-top:.5rem;display:flex;gap:.6rem;flex-wrap:wrap">
      ${['STR','DEX','CON','INT','WIS','CHA'].map((ab, i) => {
        const val = [s.strength,s.dexterity,s.constitution,s.intelligence,s.wisdom,s.charisma][i];
        return `<div style="text-align:center">
          <div style="font-size:.7rem;color:var(--text-dim)">${ab}</div>
          <div style="font-size:1rem;color:var(--text-heading);font-weight:600">${val}</div>
        </div>`;
      }).join('')}
    </div>
  `;
  chatLog.appendChild(div);
  scrollToBottom();
}

function appendJournalCard(entities) {
  if (!entities || entities.length === 0) {
    addSystemMsg('No entities in the knowledge graph yet.');
    return;
  }
  const rows = entities.map(e =>
    `<tr><td style="color:var(--text-heading);font-weight:600">${e.name}</td>
         <td style="color:var(--text-dim)">${e.summary || '—'}</td></tr>`
  ).join('');
  appendInfoCard('📜 Journal', `<table style="width:100%">${rows}</table>`);
}

function appendGraphCard(entity, results) {
  if (!results || results.length === 0) {
    addSystemMsg(`No facts found for "${entity}".`);
    return;
  }
  const facts = results.map(r => `<li>${r.fact}</li>`).join('');
  appendInfoCard(`Graph: ${entity}`, `<ul style="padding-left:1.2rem">${facts}</ul>`);
}

function appendInfoCard(title, htmlContent) {
  const div = document.createElement('div');
  div.className = 'info-card';
  div.innerHTML = `<h3>${title}</h3>${htmlContent}`;
  chatLog.appendChild(div);
  scrollToBottom();
}

/* ═══════════════════════════════════════════════════════════════
   PERSONALITY SWITCH MODAL
═══════════════════════════════════════════════════════════════ */
switchBtn.addEventListener('click', () => {
  buildPersonalityGrid(modalGrid, state.personalities, (p) => {
    modalGrid.querySelectorAll('.p-card').forEach(el =>
      el.classList.toggle('selected', el.dataset.name === p.name)
    );
    runCommand({ command: 'personality', personality: p.name });
    personalityModal.classList.add('hidden');
  });
  const cur = modalGrid.querySelector(`[data-name="${personalityLabel.textContent}"]`);
  if (cur) cur.classList.add('selected');
  personalityModal.classList.remove('hidden');
});
modalCancel.addEventListener('click', () => personalityModal.classList.add('hidden'));
personalityModal.addEventListener('click', e => {
  if (e.target === personalityModal) personalityModal.classList.add('hidden');
});

/* ═══════════════════════════════════════════════════════════════
   INPUT MODAL (roll / rules prompt)
═══════════════════════════════════════════════════════════════ */
let _inputModalCallback = null;

function promptModal(title, placeholder, callback) {
  inputModalTitle.textContent = title;
  inputModalField.value = '';
  inputModalField.placeholder = placeholder;
  _inputModalCallback = callback;
  inputModal.classList.remove('hidden');
  inputModalField.focus();
}

function submitInputModal() {
  const val = inputModalField.value.trim();
  if (val && _inputModalCallback) _inputModalCallback(val);
  inputModal.classList.add('hidden');
  _inputModalCallback = null;
}

inputModalConfirm.addEventListener('click', submitInputModal);
inputModalCancel.addEventListener('click', () => { inputModal.classList.add('hidden'); _inputModalCallback = null; });
inputModalField.addEventListener('keydown', e => { if (e.key === 'Enter') submitInputModal(); });
inputModal.addEventListener('click', e => { if (e.target === inputModal) { inputModal.classList.add('hidden'); _inputModalCallback = null; } });

/* ═══════════════════════════════════════════════════════════════
   SCENE IMAGE (Phase 13 hook)
═══════════════════════════════════════════════════════════════ */
function showSceneImage(url, caption) {
  sceneImage.classList.remove('visible');
  sceneImage.onload = () => sceneImage.classList.add('visible');
  sceneImage.src = url;
  imageCaption.textContent = caption;
}

/* ═══════════════════════════════════════════════════════════════
   MINIMAL MARKDOWN RENDERER
   Handles the subset produced by the DM: bold, italic, headers,
   unordered lists, horizontal rules, paragraphs.
═══════════════════════════════════════════════════════════════ */
function renderMarkdown(text) {
  // Escape HTML first
  let html = text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');

  // Headers
  html = html.replace(/^### (.+)$/gm, '<h3>$1</h3>');
  html = html.replace(/^## (.+)$/gm,  '<h2>$1</h2>');
  html = html.replace(/^# (.+)$/gm,   '<h1>$1</h1>');

  // Bold + italic
  html = html.replace(/\*\*\*(.+?)\*\*\*/g, '<strong><em>$1</em></strong>');
  html = html.replace(/\*\*(.+?)\*\*/g,     '<strong>$1</strong>');
  html = html.replace(/\*(.+?)\*/g,         '<em>$1</em>');
  html = html.replace(/_(.+?)_/g,           '<em>$1</em>');

  // Inline code
  html = html.replace(/`([^`]+)`/g, '<code>$1</code>');

  // HR
  html = html.replace(/^---+$/gm, '<hr>');

  // Unordered lists
  html = html.replace(/^[*-] (.+)$/gm, '<li>$1</li>');
  html = html.replace(/(<li>.*<\/li>)/s, '<ul>$1</ul>');

  // Paragraphs — split on double newline
  const blocks = html.split(/\n{2,}/);
  html = blocks.map(block => {
    block = block.trim();
    if (!block) return '';
    if (/^<(h[1-3]|ul|ol|hr|li)/.test(block)) return block;
    // Replace remaining single newlines with <br> inside a paragraph
    return '<p>' + block.replace(/\n/g, '<br>') + '</p>';
  }).join('\n');

  return html;
}

/* ═══════════════════════════════════════════════════════════════
   THREE.JS DM AVATAR
   A small procedural blocky figure whose colour scheme reflects
   the active DM personality.  Slow idle bounce via GSAP.
═══════════════════════════════════════════════════════════════ */

// Personality → [headHue, bodyHue] in HSL degrees
const PERSONALITY_COLORS = {
  'The Sage':        [45,  200],   // amber head, cool blue body
  'The Chronicler':  [210, 230],   // navy head, deep blue body
  'The Bard':        [280, 300],   // violet head, purple body
  'The Tactician':   [200, 210],   // steel blue head + body
  'The Warden':      [0,   15],    // crimson head, dark red body
  'The Mentor':      [150, 160],   // teal head, green body
};

function hsl(h, s, l) {
  return new THREE.Color(`hsl(${h}, ${s}%, ${l}%)`);
}

class DmAvatar {
  constructor(canvas, personalityData) {
    this._canvas = canvas;
    this._params = { y: 0, ry: 0, armRot: 0 };
    this._arms = [];
    this._gsapTween = null;

    this._initScene();
    this._buildFigure(personalityData);
    this._startAnimation();
  }

  _initScene() {
    const w = this._canvas.offsetWidth  || 180;
    const h = this._canvas.offsetHeight || 200;

    this._scene = new THREE.Scene();
    this._scene.background = new THREE.Color('#0d0d14');

    this._camera = new THREE.PerspectiveCamera(50, w / h, 0.1, 100);
    this._camera.position.set(0, 0.5, 5.5);

    this._renderer = new THREE.WebGLRenderer({ canvas: this._canvas, antialias: true });
    this._renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    this._renderer.setSize(w, h);

    // Lights
    const ambient = new THREE.AmbientLight(0x9090c0, 0.6);
    this._scene.add(ambient);
    const sun = new THREE.DirectionalLight(0xffffff, 1.2);
    sun.position.set(3, 5, 4);
    this._scene.add(sun);
  }

  _buildFigure(personalityData) {
    // Remove old figure if switching
    if (this._group) this._scene.remove(this._group);
    this._arms = [];
    this._group = new THREE.Group();
    this._scene.add(this._group);

    const name = personalityData ? personalityData.name : 'The Sage';
    const [hh, bh] = PERSONALITY_COLORS[name] || [45, 200];

    const headMat = new THREE.MeshLambertMaterial({ color: hsl(hh, 55, 45) });
    const bodyMat = new THREE.MeshLambertMaterial({ color: hsl(bh, 60, 38) });
    const eyeMat  = new THREE.MeshLambertMaterial({ color: hsl(hh, 20, 15) });

    const deg = d => d * Math.PI / 180;

    // Body
    const bodyGroup = new THREE.Group();
    const body = new THREE.Mesh(new THREE.BoxGeometry(1, 1.4, 0.75), bodyMat);
    bodyGroup.add(body);
    this._group.add(bodyGroup);

    // Head
    this._headGroup = new THREE.Group();
    const head = new THREE.Mesh(new THREE.BoxGeometry(1.1, 1.1, 1.0), headMat);
    this._headGroup.add(head);
    this._headGroup.position.y = 1.35;
    this._group.add(this._headGroup);

    // Eyes
    const eyeGeo = new THREE.SphereGeometry(0.1, 8, 6);
    const eyesGroup = new THREE.Group();
    eyesGroup.position.z = 0.5;
    eyesGroup.position.y = 0.08;
    for (let i = 0; i < 2; i++) {
      const eye = new THREE.Mesh(eyeGeo, eyeMat);
      eye.position.x = (i === 0 ? 1 : -1) * 0.27;
      eyesGroup.add(eye);
    }
    this._headGroup.add(eyesGroup);

    // Arms
    const armH = 0.9;
    const armGeo = new THREE.BoxGeometry(0.22, armH, 0.22);
    for (let i = 0; i < 2; i++) {
      const m = i === 0 ? 1 : -1;
      const ag = new THREE.Group();
      const arm = new THREE.Mesh(armGeo, bodyMat);
      arm.position.y = -armH * 0.5;
      ag.add(arm);
      ag.position.set(m * 0.7, 0.55, 0);
      ag.rotation.z = deg(18 * m);
      bodyGroup.add(ag);
      this._arms.push(ag);
    }

    // Legs
    const legGeo = new THREE.BoxGeometry(0.28, 0.55, 0.3);
    for (let i = 0; i < 2; i++) {
      const m = i === 0 ? 1 : -1;
      const leg = new THREE.Mesh(legGeo, bodyMat);
      leg.position.set(m * 0.25, -1.1, 0);
      bodyGroup.add(leg);
    }

    // Center figure
    this._group.position.y = -0.3;
  }

  _startAnimation() {
    if (this._gsapTween) this._gsapTween.kill();

    // Gentle idle: bob up/down + slow y-rotation + arm sway
    this._gsapTween = gsap.timeline({ repeat: -1, yoyo: true });
    this._gsapTween.to(this._params, {
      y: 0.18,
      armRot: 0.22,
      duration: 1.4,
      ease: 'sine.inOut',
    });

    gsap.to(this._params, {
      ry: Math.PI * 2,
      duration: 22,
      repeat: -1,
      ease: 'none',
    });

    // Render loop
    const tick = () => {
      this._rafId = requestAnimationFrame(tick);
      this._group.position.y = this._params.y - 0.3;
      this._group.rotation.y = this._params.ry;
      for (let i = 0; i < this._arms.length; i++) {
        const m = i === 0 ? 1 : -1;
        this._arms[i].rotation.z = (0.314 + this._params.armRot) * m;
      }
      this._renderer.render(this._scene, this._camera);
    };
    tick();
  }

  switchPersonality(personalityData) {
    this._buildFigure(personalityData);
  }

  destroy() {
    cancelAnimationFrame(this._rafId);
    if (this._gsapTween) this._gsapTween.kill();
    this._renderer.dispose();
  }
}

/* ═══════════════════════════════════════════════════════════════
   BOOT
═══════════════════════════════════════════════════════════════ */
init().catch(err => {
  setupError.textContent = `Failed to load: ${err.message}`;
  setupError.classList.remove('hidden');
});
