//! Utility to detect changed files for shader hot-reloading.

use async_channel::Receiver;
use notify::{Event, EventKind};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

#[cfg(not(target_family = "wasm"))]
use notify::Watcher;

#[cfg(doc)]
use crate::Shader;

/// State for tracking file changes.
pub struct HotReloadState {
    #[cfg(not(target_family = "wasm"))]
    watcher: notify::RecommendedWatcher,
    rcv: Receiver<notify::Result<Event>>,
    file_changed: HashMap<PathBuf, bool>,
}

impl HotReloadState {
    /// Initializes the file-tracking context.
    ///
    /// To register a shader for change-tracking call [`Shader::watch_sources`] once with the state
    /// returned by this function.
    /// To register a file for change-tracking, call [`HotReloadState::watch_file`].
    pub fn new() -> notify::Result<Self> {
        let (_snd, _rcv) = async_channel::unbounded();
        Ok(Self {
            #[cfg(not(target_family = "wasm"))]
            watcher: notify::recommended_watcher(move |msg| {
                // TODO: does hot-reloading make sense on wasm anyway?
                let _ = _snd.send_blocking(msg);
            })?,
            rcv: _rcv,
            file_changed: Default::default(),
        })
    }

    /// Saves in `self` the set of watched files that changed since the last time this function
    /// was called.
    ///
    /// Once this call completes, the [`Self::file_changed`] method can be used to check if a
    /// particular file (assuming it was added to the watch list with [`Self::watch_file`]) has
    /// changed since the last time [`Self::update_changes`] was called.
    pub fn update_changes(&mut self) {
        for changed in self.file_changed.values_mut() {
            *changed = false;
        }

        while let Ok(event) = self.rcv.try_recv() {
            if let Ok(event) = event {
                if event.need_rescan() || matches!(event.kind, EventKind::Modify(_)) {
                    for path in event.paths {
                        self.file_changed.insert(path, true);
                    }
                }
            }
        }
    }

    /// Registers a files for change-tracking.
    pub fn watch_file(&mut self, _path: &Path) -> notify::Result<()> {
        #[cfg(not(target_family = "wasm"))]
        if !self.file_changed.contains_key(_path) {
            self.watcher
                .watch(_path, notify::RecursiveMode::NonRecursive)?;
            // NOTE: this won’t insert if the watch failed.
            self.file_changed.insert(_path.to_path_buf(), false);
        }

        Ok(())
    }

    /// Checks if the specified file change was detected at the time of calling [`Self::update_changes`].
    pub fn file_changed(&self, path: &Path) -> bool {
        self.file_changed.get(path).copied().unwrap_or_default()
    }

    /// Gets the list of files watched for hot-reloading, as well as there last known change status.
    pub fn watched_files(&self) -> impl Iterator<Item = (&PathBuf, bool)> {
        self.file_changed
            .iter()
            .map(|(path, changed)| (path, *changed))
    }
}
