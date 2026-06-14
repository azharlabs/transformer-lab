import { create } from 'zustand'
import { defaultParams } from './lib/blocks.js'
import { ARCHITECTURES } from './lib/architectures.js'

let _uid = 1
const nextUid = () => `b${_uid++}`

function instantiate(key, params) {
  return { uid: nextUid(), key, params: { ...defaultParams(key), ...(params || {}) } }
}

function buildStack(spec) {
  return spec.map((s) => instantiate(s.key, s.params))
}

const DEFAULT_CONFIG = {
  modelName: 'MyTransformer',
  dModel: 256,
  vocabSize: 32000,
  maxSeqLen: 512,
  numClasses: 2,
  dropout: 0.1,
  layers: 1,
}

const STORAGE_KEY = 'transformer-lab/v1'

// Re-key a loaded stack with fresh uids so the counter never collides.
function rekey(stack) {
  return stack.map((b) => ({ uid: nextUid(), key: b.key, params: { ...b.params } }))
}

function loadPersisted() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    if (!raw) return null
    const data = JSON.parse(raw)
    if (!data || !Array.isArray(data.stack)) return null
    return { config: { ...DEFAULT_CONFIG, ...data.config }, stack: rekey(data.stack) }
  } catch {
    return null
  }
}

const persisted = typeof localStorage !== 'undefined' ? loadPersisted() : null

export const useLab = create((set, get) => ({
  view: 'builder',
  setView: (view) => set({ view }),

  config: persisted ? persisted.config : { ...DEFAULT_CONFIG },
  setConfig: (patch) => set((s) => ({ config: { ...s.config, ...patch } })),

  stack: persisted ? persisted.stack : buildStack(ARCHITECTURES['decoder'].spec),
  selectedUid: null,
  select: (uid) => set({ selectedUid: uid }),

  addBlock: (key, atIndex) =>
    set((s) => {
      const blk = instantiate(key)
      const stack = [...s.stack]
      if (atIndex == null || atIndex > stack.length) stack.push(blk)
      else stack.splice(atIndex, 0, blk)
      return { stack, selectedUid: blk.uid }
    }),

  removeBlock: (uid) =>
    set((s) => ({
      stack: s.stack.filter((b) => b.uid !== uid),
      selectedUid: s.selectedUid === uid ? null : s.selectedUid,
    })),

  moveBlock: (from, to) =>
    set((s) => {
      if (to < 0 || to >= s.stack.length || from === to) return {}
      const stack = [...s.stack]
      const [m] = stack.splice(from, 1)
      stack.splice(to, 0, m)
      return { stack }
    }),

  setBlockParam: (uid, name, value) =>
    set((s) => ({
      stack: s.stack.map((b) =>
        b.uid === uid ? { ...b, params: { ...b.params, [name]: value } } : b
      ),
    })),

  clearStack: () => set({ stack: [], selectedUid: null }),

  loadPreset: (id) =>
    set(() => ({
      stack: buildStack(ARCHITECTURES[id].spec),
      selectedUid: null,
      config: { ...get().config, ...(ARCHITECTURES[id].config || {}), layers: 1 },
    })),

  // ---- save / load / share ----
  exportJSON: () => {
    const { config, stack } = get()
    return JSON.stringify(
      { version: 1, config, stack: stack.map(({ key, params }) => ({ key, params })) },
      null,
      2
    )
  },

  importJSON: (text) => {
    try {
      const data = JSON.parse(text)
      if (!data || !Array.isArray(data.stack)) throw new Error('bad file')
      set({
        config: { ...DEFAULT_CONFIG, ...(data.config || {}) },
        stack: rekey(data.stack),
        selectedUid: null,
      })
      return true
    } catch {
      return false
    }
  },

  resetDefault: () =>
    set({
      config: { ...DEFAULT_CONFIG },
      stack: buildStack(ARCHITECTURES['decoder'].spec),
      selectedUid: null,
    }),
}))

// Persist (debounced) on every change to config / stack.
let _t = null
useLab.subscribe((s) => {
  if (typeof localStorage === 'undefined') return
  clearTimeout(_t)
  _t = setTimeout(() => {
    try {
      localStorage.setItem(
        STORAGE_KEY,
        JSON.stringify({ config: s.config, stack: s.stack.map(({ key, params }) => ({ key, params })) })
      )
    } catch {
      /* storage full or unavailable — ignore */
    }
  }, 300)
})

export { instantiate }
