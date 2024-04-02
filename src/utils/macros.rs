//! Internal helper macros and functions used therein. Not really intended for external use

/// Helper function used solely for getting the fully qualified type name of the
/// containing function
///
/// # TODO
/// Could be `const` once the following are `const` for `stable branch
///
/// * [`core::any::type_name`]
/// * [`core::slice::SliceIndex<str>`](core::slice::SliceIndex<str>) for
/// [`core::ops::Range<usize>`]
#[doc(hidden)]
#[inline(always)]
pub fn get_function_name<T>(_: T) -> &'static str {
    let s = core::any::type_name::<T>();
    &s[..s.len() - 3]
}

#[macro_export]
#[doc(hidden)]
macro_rules! fn_name {
    () => {{
        /// Helper function used solely for getting the fully qualified type name of the
        /// containing function
        const fn f() {}

        $crate::utils::macros::get_function_name(f)
    }};
}
