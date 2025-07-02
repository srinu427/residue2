use gamert::{start_window_event_loop, Game};

fn main() {
    let mut game = Game::new();
    let window_event_loop = start_window_event_loop().unwrap();
    window_event_loop.run_app(&mut game).unwrap();
}
