//! Fake Cone Module (2D Compatibility)
//!
//! This module exists solely as a workaround for naga-oil's module system.
//! In 2D mode (dim2), cones don't exist, but the shape enumeration system
//! still needs to import the cone module for conditional compilation to work.
//!
//! This empty module provides the import path without defining any actual cone
//! functionality, allowing the 2D build to compile successfully.

#define_import_path wgparry::cone
