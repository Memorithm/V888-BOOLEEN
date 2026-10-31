use std::fs;
use std::io;
use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};
use v888_booleen::connectome::Connectome;

static NEXT: AtomicUsize = AtomicUsize::new(0);

fn temp_file(extension: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "v888-learning-{}-{}.{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed),
        extension
    ))
}

fn call(args: &[&str]) -> io::Result<(bool, String, String)> {
    let output = Command::new(env!("CARGO_BIN_EXE_v888-booleen"))
        .args(args)
        .output()?;
    Ok((
        output.status.success(),
        String::from_utf8_lossy(&output.stdout).into_owned(),
        String::from_utf8_lossy(&output.stderr).into_owned(),
    ))
}

#[test]
fn train_save_score_infer_and_reject_bad_checkpoint() -> io::Result<()> {
    let graph_path = temp_file("v888b");
    let episodes_path = temp_file("csv");
    let weights_path = temp_file("v888pls");
    let bad_graph_path = temp_file("v888b");
    let original = Connectome {
        ids: vec![10, 20, 30],
        offsets: vec![0, 1, 2, 2],
        targets: vec![2, 2],
        counts: vec![1, 1],
    };
    original.save(&graph_path)?;
    fs::write(
        &episodes_path,
        "source_id,target_id,fire\n10,30,1\n20,30,0\n",
    )?;
    let g = graph_path.to_str().unwrap();
    let ep = episodes_path.to_str().unwrap();
    let w = weights_path.to_str().unwrap();
    let bad = bad_graph_path.to_str().unwrap();
    let (ok, stdout, stderr) = call(&[
        "train-plasticity",
        "--graph",
        g,
        "--episodes",
        ep,
        "--epochs",
        "4",
        "--output",
        w,
    ])?;
    assert!(ok, "{stderr}");
    assert!(stdout.contains("plasticity_training_episodes=8"));
    assert!(weights_path.exists());
    let (ok, stdout, stderr) = call(&[
        "score-plasticity",
        "--graph",
        g,
        "--weights",
        w,
        "--episodes",
        ep,
    ])?;
    assert!(ok, "{stderr}");
    assert!(stdout.contains("plasticity_correct=2 plasticity_examples=2"));
    let (ok, stdout, stderr) = call(&[
        "run-plasticity",
        "--graph",
        g,
        "--weights",
        w,
        "--seed",
        "10",
    ])?;
    assert!(ok, "{stderr}");
    assert!(stdout.contains("plasticity_active=1"));
    let (ok, _, _) = call(&[
        "train-plasticity",
        "--graph",
        g,
        "--episodes",
        ep,
        "--epochs",
        "4",
        "--output",
        w,
    ])?;
    assert!(!ok, "checkpoint must not be overwritten");
    let mut changed = original;
    changed.ids[2] = 31;
    changed.save(&bad_graph_path)?;
    let (ok, _, _) = call(&[
        "score-plasticity",
        "--graph",
        bad,
        "--weights",
        w,
        "--episodes",
        ep,
    ])?;
    assert!(!ok, "graph fingerprint must reject wrong graph");
    for path in [graph_path, episodes_path, weights_path, bad_graph_path] {
        fs::remove_file(path)?;
    }
    Ok(())
}
