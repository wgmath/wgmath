//! Extensions over naga-oil’s Composer.

use naga_oil::compose::{
    ComposableModuleDefinition, ComposableModuleDescriptor, Composer, ComposerError,
};

/// An extension trait for the naga-oil `Composer` to work around some of its limitations.
pub trait ComposerExt {
    /// Adds a composable module to `self` only if it hasn’t been added yet.
    ///
    /// Currently, `naga-oil` behaves strangely (some symbols stop resolving) if the same module is
    /// added twice. This function checks if the module has already been added. If it was already
    /// added, then `self` is left unchanged and `Ok(None)` is returned.
    fn add_composable_module_once(
        &mut self,
        desc: ComposableModuleDescriptor<'_>,
    ) -> Result<Option<&ComposableModuleDefinition>, ComposerError>;
}

impl ComposerExt for Composer {
    fn add_composable_module_once(
        &mut self,
        desc: ComposableModuleDescriptor<'_>,
    ) -> Result<Option<&ComposableModuleDefinition>, ComposerError> {
        // NOTE: extract the module name manually for avoiding duplicate. This is **much** faster
        //       than retrieving the name through `Preprocessor::get_preprocessor_metadata`.
        let module_name = desc
            .source
            .lines()
            .find(|line| line.contains("define_import_path"))
            .map(|line| {
                line.replace("#define_import_path", "")
                    .replace(";", "")
                    .trim()
                    .to_string()
            });

        if let Some(name) = &module_name {
            if self.contains_module(name) {
                // Module already exists, don’t insert it.
                return Ok(None);
            }
        }

        self.add_composable_module(desc).map(Some)
    }
}
