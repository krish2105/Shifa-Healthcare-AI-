"use client";

import { create } from "zustand";
import { persist } from "zustand/middleware";

/**
 * Effects state.
 *
 * `effects` is a manual override that is deliberately tri-state: `null` means
 * "follow the OS", true/false mean the user has decided. Collapsing this to a
 * boolean would force a default that either overrides `prefers-reduced-motion`
 * for people who set it, or disables motion for everyone else. The resolved
 * value is computed in `useEffectsEnabled`, which is the only thing components
 * should read.
 */
interface EffectsState {
  effects: boolean | null;
  setEffects: (v: boolean | null) => void;
  toggleEffects: (systemReduced: boolean) => void;
}

export const useEffectsStore = create<EffectsState>()(
  persist(
    (set, get) => ({
      effects: null,
      setEffects: (v) => set({ effects: v }),
      toggleEffects: (systemReduced) => {
        const current = get().effects ?? !systemReduced;
        set({ effects: !current });
      },
    }),
    { name: "shifa42-effects" },
  ),
);

interface SessionState {
  lastRunId: string | null;
  patientId: string | null;
  setPatientId: (v: string | null) => void;
  setLastRunId: (v: string | null) => void;
}

export const useSessionStore = create<SessionState>((set) => ({
  lastRunId: null,
  patientId: null,
  setPatientId: (v) => set({ patientId: v }),
  setLastRunId: (v) => set({ lastRunId: v }),
}));
