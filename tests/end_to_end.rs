use std::fs;
use std::io::{self,ErrorKind};
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize,Ordering};
use v888_booleen::connectome::Connectome;
use v888_booleen::engine::BooleanEngine;
static NEXT:AtomicUsize=AtomicUsize::new(0);
fn file(s:&str)->PathBuf{
    std::env::temp_dir().join(format!("v888-{}-{}-{s}",std::process::id(),NEXT.fetch_add(1,Ordering::Relaxed)))
}
fn fixture()->io::Result<(PathBuf,PathBuf,PathBuf)>{
    let edges=file("edges.csv");let ids=file("ids.txt");let binary=file("graph.v888b");
    fs::write(&edges,"pre,post,count,norm\n30,40,2,0.2\n20,30,1,0.2\n20,30,1,0.2\n40,20,4,0.5\n20,20,2,0.1\n")?;
    fs::write(&ids,"id\n10\n20\n30\n40\n50\n")?;
    Ok((edges,ids,binary))
}
#[test]fn exact_graph_roundtrip()->io::Result<()>{
    let (edge,ids,bin)=fixture()?;
    let graph=Connectome::import(&edge,Some(&ids))?;
    assert_eq!(graph.ids,[10,20,30,40,50]);
    assert_eq!(graph.offsets,[0,0,3,4,5,5]);
    assert_eq!(graph.edges(),5);
    graph.save(&bin)?;
    assert_eq!(Connectome::load(&bin)?,graph);
    let output=file("export.csv");
    graph.export_csv(&output)?;
    assert_eq!(Connectome::import(&output,Some(&ids))?,graph);
    for p in [&edge,&ids,&bin,&output]{fs::remove_file(p)?;}Ok(())
}
#[test]fn recurrent_threshold_autapse_and_multiplicity()->io::Result<()>{
    let (edge,ids,bin)=fixture()?;
    let graph=Connectome::import(&edge,Some(&ids))?;
    let mut engine=BooleanEngine::new(&graph,2)?;
    engine.stimulate(&graph,20)?;
    assert_eq!(engine.step(&graph)?,2);
    assert_eq!(engine.active.get(1),Ok(true));
    assert_eq!(engine.active.get(2),Ok(true));
    assert_eq!(engine.step(&graph)?,3);
    assert_eq!(engine.active.get(3),Ok(true));
    for p in [&edge,&ids]{fs::remove_file(p)?;}
    let _=bin;Ok(())
}
#[test]fn malformed_csv_rejected()->io::Result<()>{
    let p=file("bad.csv");
    fs::write(&p,"pre,post,count\n1,2,oops\n")?;
    assert_eq!(Connectome::import(&p,None).unwrap_err().kind(),ErrorKind::InvalidData);
    fs::remove_file(p)?;Ok(())
}
#[test]fn invalid_binary_size_rejected()->io::Result<()>{
    let p=file("bad.v888b");fs::write(&p,b"V888BOL1")?;
    assert!(Connectome::load(&p).is_err());
    fs::remove_file(p)?;Ok(())
}
