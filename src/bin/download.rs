use RBE_LLM::sllm::model_downloader::ModelDownloader;
use anyhow::Result;

#[tokio::main]
async fn main() -> Result<()> {
    let downloader = ModelDownloader::new("skt/kogpt2-base-v2");
    downloader.download().await?;
    Ok(())
} 