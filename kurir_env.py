"""Environment Kurir Paket Cerdas (grid 4x4) - Python murni, tanpa Gymnasium."""
import numpy as np

class KurirEnv:
    ATAS, BAWAH, KIRI, KANAN, AMBIL, ANTAR = range(6)
    NAMA_AKSI = ["Atas", "Bawah", "Kiri", "Kanan", "Ambil", "Antar"]
    SIMBOL = {0: "↑", 1: "↓", 2: "←", 3: "→", 4: "A", 5: "T"}

    def __init__(self):
        self.n_baris = self.n_kolom = 4
        self.start, self.titik_p, self.titik_d = (0, 0), (0, 3), (3, 0)
        self.banjir = {(1, 1), (2, 2)}
        self.n_posisi = self.n_baris * self.n_kolom
        self.n_states = self.n_posisi * 2          # 16 x 2 = 32
        self.n_actions = 6
        self.reset()

    def encode(self, pos, paket):
        return (pos[0] * self.n_kolom + pos[1]) * 2 + paket

    def decode(self, s):
        p, paket = divmod(s, 2)
        return divmod(p, self.n_kolom), paket

    def reset(self):
        self.pos, self.paket = self.start, 0
        return self.encode(self.pos, self.paket)

    def step_from(self, state, action):
        """Fungsi transisi murni: (state, action) -> (new_state, reward, done)."""
        pos, paket = self.decode(state)
        r, c = pos
        reward, done = -1, False
        if action <= self.KANAN:
            dr, dc = [(-1, 0), (1, 0), (0, -1), (0, 1)][action]
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.n_baris and 0 <= nc < self.n_kolom:
                pos = (nr, nc)
                if pos in self.banjir:
                    reward, done = -20, True
            else:
                reward = -5                      # menabrak tepi peta
        elif action == self.AMBIL:
            if pos == self.titik_p and paket == 0:
                paket, reward = 1, 10
            else:
                reward = -10
        else:                                    # ANTAR
            if pos == self.titik_d and paket == 1:
                reward, done = 30, True
            else:
                reward = -10
        return self.encode(pos, paket), reward, done

    def step(self, action):
        s = self.encode(self.pos, self.paket)
        ns, reward, done = self.step_from(s, action)
        self.pos, self.paket = self.decode(ns)
        return ns, reward, done
