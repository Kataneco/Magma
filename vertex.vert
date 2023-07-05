#version 460

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec3 color;

layout(location = 0) out vec4 fragmentColor;

layout(push_constant) uniform block {
    mat4 renderMatrix;
} constants;

void main() {
    gl_Position = constants.renderMatrix * vec4(position, 1.0f);
    fragmentColor = vec4(normal, 0.0f);
}
