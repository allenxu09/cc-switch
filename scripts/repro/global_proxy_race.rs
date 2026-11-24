// Reproduces the global proxy update race with the implementation from 148ca40a7.
// Run: rustc --edition=2021 -O scripts/repro/global_proxy_race.rs -o /tmp/global-proxy-race
// Then: /tmp/global-proxy-race
// Uses strings for clients. The barriers only delimit rounds.
use std::sync::{Barrier, OnceLock, RwLock};

mod log {
    macro_rules! discard {
        ($($arg:tt)*) => { let _ = format_args!($($arg)*); };
    }
    pub(crate) use discard as error;
    pub(crate) use discard as info;
}

type Client = Option<String>;
static GLOBAL_CLIENT: OnceLock<RwLock<Client>> = OnceLock::new();
static NO_REDIRECT_CLIENT: OnceLock<RwLock<Client>> = OnceLock::new();
static HTTPS_ONLY_CLIENT: OnceLock<RwLock<Client>> = OnceLock::new();
static CURRENT_PROXY_URL: OnceLock<RwLock<Option<String>>> = OnceLock::new();

fn build_client(proxy_url: Option<&str>) -> Result<Client, String> {
    Ok(proxy_url.map(str::to_owned))
}

use build_client as build_no_redirect_client;
use build_client as build_https_only_client;

fn init(_: Option<&str>) -> Result<(), String> {
    unreachable!("the test initializes the client before starting either update")
}

fn mask_url(url: &str) -> String {
    url.to_owned()
}

pub fn apply_proxy(proxy_url: Option<&str>) -> Result<(), String> {
    let effective_url = proxy_url.filter(|s| !s.trim().is_empty());
    let new_client = build_client(effective_url)?;
    let new_no_redirect = build_no_redirect_client(effective_url)?;
    let new_https_only = build_https_only_client(effective_url)?;

    // 更新客户端
    if let Some(lock) = GLOBAL_CLIENT.get() {
        let mut client = lock.write().map_err(|e| {
            log::error!("[GlobalProxy] [GP-001] Failed to acquire write lock: {e}");
            "Failed to update proxy: lock poisoned".to_string()
        })?;
        *client = new_client;
    } else {
        // 如果还没初始化，则初始化
        return init(proxy_url);
    }

    set_or_replace_client(&NO_REDIRECT_CLIENT, new_no_redirect)?;
    set_or_replace_client(&HTTPS_ONLY_CLIENT, new_https_only)?;

    // 更新代理 URL 记录
    if let Some(lock) = CURRENT_PROXY_URL.get() {
        let mut url = lock.write().map_err(|e| {
            log::error!("[GlobalProxy] [GP-002] Failed to acquire URL write lock: {e}");
            "Failed to update proxy URL record: lock poisoned".to_string()
        })?;
        *url = effective_url.map(|s| s.to_string());
    }

    log::info!(
        "[GlobalProxy] Applied: {}",
        effective_url
            .map(mask_url)
            .unwrap_or_else(|| "direct connection".to_string())
    );

    Ok(())
}

fn set_or_replace_client(slot: &OnceLock<RwLock<Client>>, client: Client) -> Result<(), String> {
    if let Some(lock) = slot.get() {
        let mut guard = lock
            .write()
            .map_err(|_| "Failed to update HTTP client: lock poisoned".to_string())?;
        *guard = client;
    } else if let Err(existing) = slot.set(RwLock::new(client)) {
        // 并发初始化竞争：改写已存在的槽位
        if let Some(lock) = slot.get() {
            let mut guard = lock
                .write()
                .map_err(|_| "Failed to update HTTP client: lock poisoned".to_string())?;
            *guard = existing
                .into_inner()
                .map_err(|_| "Failed to update HTTP client: lock poisoned".to_string())?;
        }
    }
    Ok(())
}

fn main() {
    GLOBAL_CLIENT.set(RwLock::new(None)).unwrap();
    CURRENT_PROXY_URL.set(RwLock::new(None)).unwrap();

    const ROUNDS: usize = 100_000;
    const PROXIES: [&str; 8] = ["A", "B", "C", "D", "E", "F", "G", "H"];
    let start = Barrier::new(PROXIES.len() + 1);
    let finish = Barrier::new(PROXIES.len() + 1);
    let mut mismatches = 0;

    std::thread::scope(|scope| {
        for proxy in PROXIES {
            let start = &start;
            let finish = &finish;
            scope.spawn(move || {
                for _ in 0..ROUNDS {
                    start.wait();
                    apply_proxy(Some(proxy)).unwrap();
                    finish.wait();
                }
            });
        }
        for round in 1..=ROUNDS {
            start.wait();
            finish.wait();
            let primary = GLOBAL_CLIENT.get().unwrap().read().unwrap();
            let url = CURRENT_PROXY_URL.get().unwrap().read().unwrap();
            if *primary != *url {
                mismatches += 1;
                if mismatches == 1 {
                    println!("round {round}: primary={primary:?}, URL={url:?}");
                }
            }
        }
    });
    println!("{mismatches}/{ROUNDS} rounds ended with different proxy values");
}
