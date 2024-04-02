#version 450

layout(push_constant) uniform UBO {
	mat4 transform;
	vec2 tex_offset;
	vec2 tex_extent;
} data;

layout(location = 0) out vec2 tex_pos;

vec2 verts[4] = vec2[](
	vec2(0.0, 0.0),
	vec2(1.0, 0.0),
	vec2(0.0, 1.0),
	vec2(1.0, 1.0)
);

void main() {
	// vec2 pos = verts[gl_VertexIndex % 4];
	vec2 pos = vec2(
		float((gl_VertexIndex + 1) & 2) * 0.5f,
		float(gl_VertexIndex & 2) * 0.5f
	);
	tex_pos = (pos * data.tex_extent) + data.tex_offset;

	gl_Position = data.transform * vec4(pos, 0.0, 1.0);
}
