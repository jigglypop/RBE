use hf_hub::{api::tokio::Api, Repo, RepoType};
use std::path::PathBuf;
use anyhow::Result;

#[tokio::main]
async fn main() -> Result<()> {
    let model_id = "skt/kogpt2-base-v2";
    let filename_to_download = "model.safetensors";
    let output_dir = PathBuf::from("models/skt-kogpt2-base-v2");

    println!(
        "Downloading '{}' from '{}'...",
        filename_to_download, model_id
    );

    let api = Api::new()?;
    let repo = api.repo(Repo::new(model_id.to_string(), RepoType::Model));

    let download_path = repo.get(filename_to_download).await?;

    // 다운로드된 파일을 원하는 위치로 복사/이동
    std::fs::create_dir_all(&output_dir)?;
    let final_path = output_dir.join(filename_to_download);
    std::fs::copy(&download_path, &final_path)?;

    println!(
        "✅ Successfully downloaded and moved to '{}'",
        final_path.display()
    );

    Ok(())
} 