fn main() {
    let arg = std::env::args().nth(1);
    match arg.as_deref() {
        Some("info") | None => {
            println!("V888 BOOLEEN 0.1.0 - initialisation Rust uniquement");
            println!("Aucune donnee V888 importee ; aucun simulateur ou modele entraine.");
        }
        Some("--help") | Some("-h") => {
            println!("Usage: v888-booleen [info|--help]");
        }
        Some(_) => {
            eprintln!("Commande inconnue. Essayez --help.");
            std::process::exit(2);
        }
    }
}
