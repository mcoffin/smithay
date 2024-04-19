/// Helper function used solely for getting the fully qualified type name of the
/// containing function
///
/// # TODO
/// Could be `const` once [`std::any::type_name`] is `const` for `stable branch
#[doc(hidden)]
#[inline(always)]
pub fn get_function_name<T>(_: T) -> &'static str {
    let s = core::any::type_name::<T>();
    s.strip_suffix("::f")
        .unwrap_or(s)
}

#[macro_export]
#[doc(hidden)]
macro_rules! fn_name {
    () => {
        {
            /// Helper function used solely for getting the fully qualified type name of the
            /// containing function
            const fn f() {}

            $crate::utils::macros::get_function_name(f)
        }
    };
}
