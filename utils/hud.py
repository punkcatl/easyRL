import pygame


_hud_state = {"episode": 0, "n_episodes": 0, "epsilon": 0.0, "reward": 0.0}


def patch_viewer_for_hud(env):
    """Monkey-patch the viewer's display method to draw HUD before flip."""
    viewer = getattr(env.unwrapped, "viewer", None)
    if viewer is None or not viewer.enabled:
        return

    original_display = viewer.display

    def display_with_hud():
        original_display()
        font = pygame.font.SysFont("monospace", 40, bold=True)
        text = (f"Episode: {_hud_state['episode']}/{_hud_state['n_episodes']}  "
                f"Epsilon: {_hud_state['epsilon']:.3f}  "
                f"Reward: {_hud_state['reward']:.1f}")
        surface = font.render(text, True, (255, 255, 255), (0, 0, 0))
        viewer.screen.blit(surface, (10, 10))
        pygame.display.update(surface.get_rect(topleft=(10, 10)))

    viewer.display = display_with_hud


def update_hud(episode, n_episodes, epsilon, reward):
    """Update HUD state values."""
    _hud_state["episode"] = episode
    _hud_state["n_episodes"] = n_episodes
    _hud_state["epsilon"] = epsilon
    _hud_state["reward"] = reward
