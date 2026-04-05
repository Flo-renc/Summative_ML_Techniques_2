"""
rendering.py
============
Pygame-based visualisation for the Teacher Retention Environment.

Displays a 2D dashboard showing:
  - A rural school scene with the teacher agent (animated sprite)
  - Live bar gauges for all 10 observation dimensions
  - Episode stats: months served, consecutive transfers, current action
  - Colour-coded risk indicators (green / amber / red)
  - Scrolling reward history graph
"""

import numpy as np
from typing import Optional

# Observation indices (mirrors custom_env.py)
OBS_SALARY, OBS_HOUSING, OBS_PROF_DEV, OBS_COMMUNITY = 0, 1, 2, 3
OBS_WORKLOAD, OBS_ISOLATION, OBS_RESOURCES, OBS_HEALTH = 4, 5, 6, 7
OBS_FAMILY, OBS_MONTHS = 8, 9

ACTION_LABELS = [
    "PERSEVERE",
    "SEEK SUPPORT",
    "COMMUNITY BUILD",
    "SELF CARE",
    "REQUEST TRANSFER",
]

# Colour palette (Zambian-inspired earthy tones)
C_BG         = (245, 235, 215)   # warm cream
C_PANEL      = (55,  75,  55)    # dark green
C_PANEL_LITE = (80, 110, 80)     # lighter green
C_TEXT       = (255, 248, 230)   # off-white
C_TEXT_DARK  = (40,  30,  20)    # near-black
C_ACCENT     = (220, 160, 40)    # golden yellow
C_GOOD       = (80,  200, 100)   # healthy green
C_WARN       = (240, 180, 50)    # amber
C_BAD        = (220, 70,  60)    # red
C_NEUTRAL    = (150, 190, 220)   # soft blue
C_SKY        = (135, 195, 235)
C_EARTH      = (160, 120, 70)
C_GRASS      = (100, 160, 80)
C_SCHOOL     = (220, 200, 160)


class TeacherRetentionRenderer:
    """
    Pygame renderer for TeacherRetentionEnv.

    Parameters
    ----------
    headless : bool
        If True, uses offscreen surface (for rgb_array mode without display).
    """

    WIDTH  = 900
    HEIGHT = 580

    def __init__(self, headless: bool = False):
        import pygame
        self.pygame = pygame
        self._headless = headless

        pygame.init()
        pygame.display.set_caption("Teacher Retention – Zambia Rural School RL")

        if headless:
            import os
            os.environ.setdefault("SDL_VIDEODRIVER", "offscreen")
            self._screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        else:
            self._screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))

        self._clock  = pygame.font.init() or None
        self._clock  = pygame.time.Clock()

        # Fonts
        self._font_lg = pygame.font.SysFont("DejaVuSans", 18, bold=True)
        self._font_md = pygame.font.SysFont("DejaVuSans", 14)
        self._font_sm = pygame.font.SysFont("DejaVuSans", 11)

        # Reward history buffer
        self._reward_history: list = []

        # Agent animation
        self._anim_tick = 0

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def render(self, state: np.ndarray, months: int, info: dict):
        pg = self.pygame
        screen = self._screen

        # Handle quit events
        for event in pg.event.get():
            if event.type == pg.QUIT:
                self.close()
                return

        self._anim_tick += 1

        # Track reward (from info if available)
        reward = info.get("last_reward", 0.0)
        self._reward_history.append(reward)
        if len(self._reward_history) > 60:
            self._reward_history.pop(0)

        # ── Draw layers ──────────────────────────────────────────────────
        self._draw_scene(screen, state, months, info)
        self._draw_gauges(screen, state)
        self._draw_stats(screen, state, months, info)
        self._draw_reward_graph(screen)

        pg.display.flip()
        self._clock.tick(self.metadata_fps if hasattr(self, "metadata_fps") else 4)

    def get_rgb_array(self, state: np.ndarray, months: int) -> np.ndarray:
        """Return current frame as RGB numpy array."""
        self.render(state, months, {})
        pg = self.pygame
        return pg.surfarray.array3d(self._screen).transpose(1, 0, 2)

    def close(self):
        try:
            self.pygame.quit()
        except Exception:
            pass

    # -----------------------------------------------------------------------
    # Drawing helpers
    # -----------------------------------------------------------------------

    def _draw_scene(self, screen, state, months, info):
        pg = self.pygame

        # Background sky
        screen.fill(C_SKY)

        # Ground / earth strip
        pg.draw.rect(screen, C_EARTH, (0, 320, self.WIDTH, self.HEIGHT - 320))

        # Grass
        pg.draw.rect(screen, C_GRASS, (0, 310, self.WIDTH, 30))

        # ── School building ───────────────────────────────────────────────
        self._draw_school(screen, 80, 200)

        # ── Trees ─────────────────────────────────────────────────────────
        for tx in [30, 280, 320, 370]:
            self._draw_tree(screen, tx, 295)

        # ── Agent (teacher) ───────────────────────────────────────────────
        bob = int(3 * np.sin(self._anim_tick * 0.3))  # gentle bobbing
        self._draw_agent(screen, 220, 270 + bob, state)

        # ── Episode header ────────────────────────────────────────────────
        month_text = self._font_lg.render(
            f"Month {months} / 24", True, C_TEXT_DARK
        )
        screen.blit(month_text, (10, 10))

        # Action label
        last_action = info.get("last_action", None)
        if last_action is not None:
            label = ACTION_LABELS[last_action]
            colour = C_WARN if last_action == 4 else C_ACCENT
            action_surf = self._font_md.render(f"Action: {label}", True, colour)
            screen.blit(action_surf, (10, 35))

        # Termination reason
        reason = info.get("termination_reason")
        if reason:
            r_col = C_GOOD if "SUCCESS" in reason else C_BAD
            r_surf = self._font_lg.render(reason, True, r_col)
            screen.blit(r_surf, (self.WIDTH // 2 - r_surf.get_width() // 2, 10))

    def _draw_school(self, screen, x, y):
        pg = self.pygame
        # Building body
        pg.draw.rect(screen, C_SCHOOL, (x, y, 160, 110))
        # Roof (triangle)
        pg.draw.polygon(screen, (180, 100, 60), [
            (x - 10, y), (x + 80, y - 50), (x + 170, y)
        ])
        # Door
        pg.draw.rect(screen, (100, 70, 40), (x + 65, y + 60, 30, 50))
        # Windows
        for wx in [x + 15, x + 110]:
            pg.draw.rect(screen, C_SKY, (wx, y + 20, 30, 25))
            pg.draw.line(screen, (100, 80, 40), (wx + 15, y + 20), (wx + 15, y + 45), 2)
            pg.draw.line(screen, (100, 80, 40), (wx, y + 32), (wx + 30, y + 32), 2)
        # Sign
        sign = self._font_sm.render("RURAL SCHOOL", True, C_TEXT_DARK)
        screen.blit(sign, (x + 10, y + 5))

    def _draw_tree(self, screen, x, y):
        pg = self.pygame
        pg.draw.rect(screen, (100, 70, 40), (x + 8, y, 8, 25))
        pg.draw.circle(screen, C_GRASS, (x + 12, y - 5), 18)
        pg.draw.circle(screen, (80, 140, 60), (x + 5, y + 2), 12)

    def _draw_agent(self, screen, x, y, state):
        """Draw the teacher agent — colour reflects health/wellbeing."""
        pg = self.pygame
        health = float(state[OBS_HEALTH])
        body_col = self._lerp_colour(C_BAD, C_GOOD, health)

        # Body
        pg.draw.ellipse(screen, body_col, (x - 10, y - 20, 20, 35))
        # Head
        pg.draw.circle(screen, (230, 190, 140), (x, y - 28), 12)
        # Hat (teacher)
        pg.draw.rect(screen, (50, 50, 80), (x - 10, y - 42, 20, 5))
        pg.draw.rect(screen, (50, 50, 80), (x - 7, y - 50, 14, 10))
        # Arms (waving if action == SEEK_SUPPORT)
        arm_ang = int(8 * np.sin(self._anim_tick * 0.5))
        pg.draw.line(screen, body_col, (x - 10, y - 10), (x - 20, y + arm_ang), 3)
        pg.draw.line(screen, body_col, (x + 10, y - 10), (x + 20, y - arm_ang), 3)

    def _draw_gauges(self, screen, state):
        """Right-side panel with 10 bar gauges."""
        pg = self.pygame
        PANEL_X = 420
        PANEL_W = 460
        PANEL_H = self.HEIGHT

        # Panel background
        pg.draw.rect(screen, C_PANEL, (PANEL_X, 0, PANEL_W, PANEL_H))

        title = self._font_lg.render("Observation State", True, C_ACCENT)
        screen.blit(title, (PANEL_X + 10, 8))

        labels = [
            ("Salary Index",         OBS_SALARY,    False),
            ("Housing Quality",      OBS_HOUSING,   False),
            ("Professional Dev",     OBS_PROF_DEV,  False),
            ("Community Ties",       OBS_COMMUNITY, False),
            ("Workload Pressure",    OBS_WORKLOAD,  True),   # high = bad
            ("Isolation Score",      OBS_ISOLATION, True),   # high = bad
            ("Resource Avail.",      OBS_RESOURCES, False),
            ("Health Status",        OBS_HEALTH,    False),
            ("Family Proximity",     OBS_FAMILY,    False),
            ("Months Served",        OBS_MONTHS,    False),
        ]

        BAR_X     = PANEL_X + 160
        BAR_W     = 250
        BAR_H     = 18
        START_Y   = 38
        ROW_H     = 50

        for i, (label, idx, invert) in enumerate(labels):
            y = START_Y + i * ROW_H

            val = float(state[idx])
            bar_val = val

            # Colour: invert means high value = bad
            if invert:
                col = self._lerp_colour(C_GOOD, C_BAD, val)
            else:
                col = self._lerp_colour(C_BAD, C_GOOD, val)

            # Label
            lbl_surf = self._font_md.render(label, True, C_TEXT)
            screen.blit(lbl_surf, (PANEL_X + 10, y + 1))

            # Bar background
            pg.draw.rect(screen, C_PANEL_LITE, (BAR_X, y, BAR_W, BAR_H), border_radius=4)

            # Bar fill
            fill_w = int(BAR_W * bar_val)
            if fill_w > 0:
                pg.draw.rect(screen, col, (BAR_X, y, fill_w, BAR_H), border_radius=4)

            # Value text
            val_surf = self._font_sm.render(f"{val:.2f}", True, C_TEXT)
            screen.blit(val_surf, (BAR_X + BAR_W + 6, y + 3))

            # Risk indicator dot
            if invert and val > 0.80:
                pg.draw.circle(screen, C_BAD, (PANEL_X + 150, y + 9), 5)
            elif not invert and val < 0.20:
                pg.draw.circle(screen, C_WARN, (PANEL_X + 150, y + 9), 5)

    def _draw_stats(self, screen, state, months, info):
        """Bottom-left stats strip."""
        pg = self.pygame
        y = 370
        x = 20

        stats = [
            f"Months served:       {months} / 24",
            f"Transfer requests:   {info.get('consecutive_transfers', 0)} / 3",
            f"Health:              {float(state[OBS_HEALTH]):.2f}",
            f"Workload:            {float(state[OBS_WORKLOAD]):.2f}",
        ]
        for line in stats:
            surf = self._font_md.render(line, True, C_TEXT_DARK)
            screen.blit(surf, (x, y))
            y += 22

    def _draw_reward_graph(self, screen):
        """Mini scrolling reward curve in the bottom-left corner."""
        pg = self.pygame
        if len(self._reward_history) < 2:
            return

        GX, GY, GW, GH = 20, 470, 380, 90
        pg.draw.rect(screen, (200, 190, 175), (GX, GY, GW, GH), border_radius=4)
        pg.draw.rect(screen, C_TEXT_DARK, (GX, GY, GW, GH), 1, border_radius=4)

        title = self._font_sm.render("Reward History", True, C_TEXT_DARK)
        screen.blit(title, (GX + 4, GY + 2))

        hist = self._reward_history[-GW:]
        mn, mx = min(hist) - 0.5, max(hist) + 0.5
        rng = mx - mn if mx != mn else 1.0

        pts = []
        for i, r in enumerate(hist):
            px = GX + int(i * GW / max(len(hist), 1))
            py = GY + GH - int((r - mn) / rng * (GH - 20)) - 5
            pts.append((px, py))

        if len(pts) >= 2:
            pg.draw.lines(screen, C_PANEL, False, pts, 2)

        # Zero line
        zero_y = GY + GH - int((0 - mn) / rng * (GH - 20)) - 5
        pg.draw.line(screen, C_WARN, (GX, zero_y), (GX + GW, zero_y), 1)

    # -----------------------------------------------------------------------
    # Utility
    # -----------------------------------------------------------------------

    @staticmethod
    def _lerp_colour(c1, c2, t):
        t = max(0.0, min(1.0, t))
        return tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))