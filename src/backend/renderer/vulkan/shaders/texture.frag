#version 450

layout(set = 0, binding = 0) uniform sampler2D tex;

layout(location = 0) in vec2 tex_pos;
layout(location = 0) out vec4 frag_color;

layout(push_constant) uniform UBO {
	layout(offset = 80) float alpha;
} data;

layout (constant_id = 0) const int COLOR_TRANSFORM = 0;

float srgb_channel_to_linear(float x) {
	return mix(x / 12.92,
		pow((x + 0.055) / 1.055, 2.4),
		x > 0.04045);
}

vec4 srgb_color_to_linear(vec4 color) {
	if (color.a == 0) {
		return vec4(0);
	}
	color.rgb /= color.a;
	color.rgb = vec3(
		srgb_channel_to_linear(color.r),
		srgb_channel_to_linear(color.g),
		srgb_channel_to_linear(color.b)
	);
	color.rgb *= color.a;
	return color;
}

void main() {
	vec4 tex_color = textureLod(tex, tex_pos, 0);
	if (COLOR_TRANSFORM == 1) {
		frag_color = srgb_color_to_linear(tex_color);
	} else {
		frag_color = tex_color;
	}
	frag_color *= data.alpha;
}
