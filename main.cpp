//C++ standard library imports
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <chrono>
#include <string>
#include <vector>
#include <unordered_map>

//Import Volk, define VK_NO_PROTOTYPES since we're not linking Vulkan statically
#define VK_NO_PROTOTYPES
#define VOLK_IMPLEMENTATION
#include <volk.h>
#include <shaderc/shaderc.hpp>

//Import Vulkan Memory Allocator, disable automatic Vulkan function fetching
#define VMA_IMPLEMENTATION
#define VMA_STATIC_VULKAN_FUNCTIONS 0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 0
#include <vk_mem_alloc.h>

//Import GLFW, don't include definitions for non Vulkan APIs
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

//Import GLM, also use it's handy countof definition
#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtx/string_cast.hpp>
#include <glm/gtx/hash.hpp>
using glm::countof;

//Import Tiny OBJ Loader
#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>

//Util: Read file into string
std::string readFile(const char* filePath) {
    std::ifstream file(filePath, std::ios::ate | std::ios::binary);

    if(!file.is_open()) throw std::runtime_error("Failed to open file");

    size_t fileSize = static_cast<size_t>(file.tellg());
    std::string data(fileSize, ' ');

    file.seekg(0);
    file.read(data.data(), static_cast<long>(fileSize));
    file.close();

    return data;
}

//Util: Simple image view creator
void createImageView(VkImageView &imageView, VkImage image, VkImageViewType viewType, VkFormat format, VkImageSubresourceRange subresourceRange, VkDevice device) {
    VkImageViewCreateInfo imageViewCreateInfo{};
    imageViewCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    imageViewCreateInfo.image = image;
    imageViewCreateInfo.viewType = viewType;
    imageViewCreateInfo.format = format;
    imageViewCreateInfo.subresourceRange = subresourceRange;

    if (vkCreateImageView(device, &imageViewCreateInfo, nullptr, &imageView) != VK_SUCCESS)
        throw std::runtime_error("Failed to create image view");
}

//Util: Compile GLSL shader into SPIR-V bytecode
std::vector<uint32_t> compileShader(const std::string& shaderSource, shaderc_shader_kind kind) {
    shaderc::Compiler compiler;
    shaderc::CompileOptions options;
    options.SetTargetEnvironment(shaderc_target_env::shaderc_target_env_vulkan, shaderc_env_version::shaderc_env_version_vulkan_1_3);
    options.SetSourceLanguage(shaderc_source_language::shaderc_source_language_glsl);
    options.SetTargetSpirv(shaderc_spirv_version::shaderc_spirv_version_1_6);
    options.SetWarningsAsErrors();
    options.SetOptimizationLevel(shaderc_optimization_level::shaderc_optimization_level_performance);

    shaderc::SpvCompilationResult module = compiler.CompileGlslToSpv(shaderSource, kind, "main", options);

    if (module.GetCompilationStatus() != shaderc_compilation_status_success) {
        throw std::runtime_error(module.GetErrorMessage());
    }

    return { module.cbegin(), module.cend() };
}

//Util: Simple shader module creator
VkShaderModule createShaderModule(std::vector<uint32_t> code, VkDevice device) {
    VkShaderModule shaderModule;
    VkShaderModuleCreateInfo shaderModuleCreateInfo{};
    shaderModuleCreateInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    shaderModuleCreateInfo.codeSize = code.size() * sizeof(uint32_t);
    shaderModuleCreateInfo.pCode = code.data();

    if (vkCreateShaderModule(device, &shaderModuleCreateInfo, nullptr, &shaderModule) != VK_SUCCESS)
        throw std::runtime_error("Failed to create shader module");
    return shaderModule;
}

//Mesh: Vertex description
struct VertexDescription {
    std::vector<VkVertexInputBindingDescription> bindings;
    std::vector<VkVertexInputAttributeDescription> attributes;
};

//Mesh: Vertex structure
struct Vertex {
    glm::vec3 position;
    glm::vec3 normal;
    glm::vec3 color;

    static VertexDescription vertexDescription();

    bool operator==(const Vertex &other) const {
        return position == other.position &&
               normal == other.normal &&
               color == other.color;
    }
};

//Mesh: Vertex description definition
VertexDescription Vertex::vertexDescription() {
    VertexDescription description;

    VkVertexInputBindingDescription binding{};
    binding.binding = 0;
    binding.stride = sizeof(Vertex);
    binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
    description.bindings.push_back(binding);

    VkVertexInputAttributeDescription positionAttribute{};
    positionAttribute.binding = 0;
    positionAttribute.location = 0;
    positionAttribute.format = VK_FORMAT_R32G32B32_SFLOAT;
    positionAttribute.offset = offsetof(Vertex, position);

    VkVertexInputAttributeDescription normalAttribute{};
    normalAttribute.binding = 0;
    normalAttribute.location = 1;
    normalAttribute.format = VK_FORMAT_R32G32B32_SFLOAT;
    normalAttribute.offset = offsetof(Vertex, normal);

    VkVertexInputAttributeDescription colorAttribute{};
    colorAttribute.binding = 0;
    colorAttribute.location = 2;
    colorAttribute.format = VK_FORMAT_R32G32B32_SFLOAT;
    colorAttribute.offset = offsetof(Vertex, color);

    description.attributes.push_back(positionAttribute);
    description.attributes.push_back(normalAttribute);
    description.attributes.push_back(colorAttribute);
    return description;
}

//Mesh: Vertex hashing
namespace std {
    template<>
    struct hash<Vertex> {
        size_t operator()(Vertex const &vertex) const {
            return ((hash<glm::vec3>()(vertex.position) ^ (hash<glm::vec3>()(vertex.normal) << 1)) >> 1) ^ (hash<glm::vec3>()(vertex.color) << 1);
        }
    };
}

//Mesh: Mesh structure
struct Mesh {
    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;

    static Mesh loadObj(const std::string& filePath);
};

//TODO: This is a testing function, replace later
Mesh Mesh::loadObj(const std::string& filePath) {
    Mesh mesh{};
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string warn, err;

    if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, filePath.c_str()))
        throw std::runtime_error(warn + err);

    std::unordered_map<Vertex, uint32_t> uniqueVertices{};

    for (const auto &shape: shapes) {
        for (const auto &index: shape.mesh.indices) {
            Vertex vertex{};

            vertex.position = {
                    attrib.vertices[3 * index.vertex_index + 0],
                    attrib.vertices[3 * index.vertex_index + 1],
                    attrib.vertices[3 * index.vertex_index + 2]
            };

            vertex.normal = {
                    attrib.normals[3 * index.vertex_index + 0],
                    attrib.normals[3 * index.vertex_index + 1],
                    attrib.normals[3 * index.vertex_index + 2]
            };

            vertex.color = {1.0f, 1.0f, 1.0f};

            if (uniqueVertices.count(vertex) == 0) {
                uniqueVertices[vertex] = static_cast<uint32_t>(mesh.vertices.size());
                mesh.vertices.push_back(vertex);
            }

            mesh.indices.push_back(uniqueVertices[vertex]);
        }
    }
    return mesh;
}

int main() {
    //Hardcoded resolution
    int width = 1280, height = 720;

    //Create GLFW window
    glfwInit();
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    GLFWwindow *window = glfwCreateWindow(width, height, "I <3 You", nullptr, nullptr);

    //Vulkan initialize
    volkInitialize();

    VkInstance instance;

    uint32_t glfwExtensionCount = 0;
    const char** glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

    std::vector<const char*> instanceExtensions(glfwExtensions, glfwExtensions + glfwExtensionCount);

    VkApplicationInfo applicationInfo{};
    applicationInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    applicationInfo.apiVersion = VK_API_VERSION_1_3;
    applicationInfo.pEngineName = "Magma Engine";
    applicationInfo.engineVersion = VK_MAKE_VERSION(0, 0, 1);
    applicationInfo.pApplicationName = "Magma Tests";
    applicationInfo.applicationVersion = VK_MAKE_VERSION(0, 0, 0);

    VkInstanceCreateInfo instanceCreateInfo{};
    instanceCreateInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instanceCreateInfo.pApplicationInfo = &applicationInfo;
    instanceCreateInfo.enabledExtensionCount = instanceExtensions.size();
    instanceCreateInfo.ppEnabledExtensionNames = instanceExtensions.data();

    if (vkCreateInstance(&instanceCreateInfo, nullptr, &instance) != VK_SUCCESS)
        throw std::runtime_error("Failed to create instance");

    volkLoadInstance(instance);

    VkPhysicalDevice physicalDevice;
    uint32_t ione = 1;
    vkEnumeratePhysicalDevices(instance, &ione, &physicalDevice);

    uint32_t graphicsFamily = 0;

    uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, nullptr);

    auto *queueFamilies = new VkQueueFamilyProperties[queueFamilyCount];
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, queueFamilies);

    for (uint32_t i = 0; i < queueFamilyCount; ++i)
        if (queueFamilies[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) graphicsFamily = i;

    delete[] queueFamilies;

    float fone = 1.0f;

    VkDeviceQueueCreateInfo graphicsQueueCreateInfo{};
    graphicsQueueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    graphicsQueueCreateInfo.queueFamilyIndex = graphicsFamily;
    graphicsQueueCreateInfo.queueCount = 1;
    graphicsQueueCreateInfo.pQueuePriorities = &fone;

    VkDeviceQueueCreateInfo queueCreateInfos[] = {graphicsQueueCreateInfo};

    VkPhysicalDeviceFeatures features{.samplerAnisotropy = 16};
    const char *extensions[] = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};

    VkDeviceCreateInfo deviceCreateInfo{};
    deviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    deviceCreateInfo.pQueueCreateInfos = queueCreateInfos;
    deviceCreateInfo.queueCreateInfoCount = countof(queueCreateInfos);
    deviceCreateInfo.pEnabledFeatures = &features;
    deviceCreateInfo.enabledExtensionCount = countof(extensions);
    deviceCreateInfo.ppEnabledExtensionNames = extensions;

    VkDevice device;
    if (vkCreateDevice(physicalDevice, &deviceCreateInfo, nullptr, &device) != VK_SUCCESS)
        throw std::runtime_error("Failed to create device");

    volkLoadDevice(device);

    //Initialize Vulkan Memory Allocator to help with memory management
    VmaVulkanFunctions vmaVulkanFunctions{};
    vmaVulkanFunctions.vkGetInstanceProcAddr = vkGetInstanceProcAddr;
    vmaVulkanFunctions.vkGetDeviceProcAddr = vkGetDeviceProcAddr;
    vmaVulkanFunctions.vkGetPhysicalDeviceProperties = vkGetPhysicalDeviceProperties;
    vmaVulkanFunctions.vkGetPhysicalDeviceMemoryProperties = vkGetPhysicalDeviceMemoryProperties;
    vmaVulkanFunctions.vkAllocateMemory = vkAllocateMemory;
    vmaVulkanFunctions.vkFreeMemory = vkFreeMemory;
    vmaVulkanFunctions.vkMapMemory = vkMapMemory;
    vmaVulkanFunctions.vkUnmapMemory = vkUnmapMemory;
    vmaVulkanFunctions.vkFlushMappedMemoryRanges = vkFlushMappedMemoryRanges;
    vmaVulkanFunctions.vkInvalidateMappedMemoryRanges = vkInvalidateMappedMemoryRanges;
    vmaVulkanFunctions.vkBindBufferMemory = vkBindBufferMemory;
    vmaVulkanFunctions.vkBindImageMemory = vkBindImageMemory;
    vmaVulkanFunctions.vkGetBufferMemoryRequirements = vkGetBufferMemoryRequirements;
    vmaVulkanFunctions.vkGetImageMemoryRequirements = vkGetImageMemoryRequirements;
    vmaVulkanFunctions.vkCreateBuffer = vkCreateBuffer;
    vmaVulkanFunctions.vkDestroyBuffer = vkDestroyBuffer;
    vmaVulkanFunctions.vkCreateImage = vkCreateImage;
    vmaVulkanFunctions.vkDestroyImage = vkDestroyImage;
    vmaVulkanFunctions.vkCmdCopyBuffer = vkCmdCopyBuffer;
    vmaVulkanFunctions.vkGetBufferMemoryRequirements2KHR = vkGetBufferMemoryRequirements2;
    vmaVulkanFunctions.vkGetImageMemoryRequirements2KHR = vkGetImageMemoryRequirements2;
    vmaVulkanFunctions.vkBindBufferMemory2KHR = vkBindBufferMemory2;
    vmaVulkanFunctions.vkBindImageMemory2KHR = vkBindImageMemory2;
    vmaVulkanFunctions.vkGetPhysicalDeviceMemoryProperties2KHR = vkGetPhysicalDeviceMemoryProperties2;
    vmaVulkanFunctions.vkGetDeviceBufferMemoryRequirements = vkGetDeviceBufferMemoryRequirements;
    vmaVulkanFunctions.vkGetDeviceImageMemoryRequirements = vkGetDeviceImageMemoryRequirements;

    VmaAllocatorCreateInfo vmaAllocatorCreateInfo{};
    vmaAllocatorCreateInfo.vulkanApiVersion = VK_API_VERSION_1_3;
    vmaAllocatorCreateInfo.instance = instance;
    vmaAllocatorCreateInfo.physicalDevice = physicalDevice;
    vmaAllocatorCreateInfo.device = device;
    vmaAllocatorCreateInfo.pVulkanFunctions = &vmaVulkanFunctions;

    VmaAllocator allocator;
    if (vmaCreateAllocator(&vmaAllocatorCreateInfo, &allocator) != VK_SUCCESS)
        throw std::runtime_error("Failed to initialize Vulkan Memory Allocator");

    //Get a graphics queue handle
    VkQueue graphicsQueue;
    vkGetDeviceQueue(device, graphicsFamily, 0, &graphicsQueue);

    //Create a Vulkan surface to interface with the GLFW window
    VkSurfaceKHR surface;
    glfwCreateWindowSurface(instance, window, nullptr, &surface);

    //Create a swapchain to hold our main framebuffer(s)
    VkSwapchainKHR swapchain = VK_NULL_HANDLE;
    VkSwapchainCreateInfoKHR swapchainCreateInfo{};
    swapchainCreateInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    swapchainCreateInfo.surface = surface;
    swapchainCreateInfo.minImageCount = 3;
    swapchainCreateInfo.imageFormat = VK_FORMAT_R8G8B8A8_SRGB;
    swapchainCreateInfo.imageColorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
    swapchainCreateInfo.imageExtent = {static_cast<uint32_t>(width), static_cast<uint32_t>(height)};
    swapchainCreateInfo.imageArrayLayers = 1;
    swapchainCreateInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    swapchainCreateInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    swapchainCreateInfo.queueFamilyIndexCount = 1;
    swapchainCreateInfo.pQueueFamilyIndices = &graphicsFamily;
    swapchainCreateInfo.preTransform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR;
    swapchainCreateInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    swapchainCreateInfo.presentMode = VK_PRESENT_MODE_MAILBOX_KHR;
    swapchainCreateInfo.clipped = VK_TRUE;
    swapchainCreateInfo.oldSwapchain = swapchain;

    if (vkCreateSwapchainKHR(device, &swapchainCreateInfo, nullptr, &swapchain))
        throw std::runtime_error("Failed to create swapchain");

    //Create an image view for each image in our swapchain to later use in framebuffer creation
    uint32_t swapchainImageCount;
    vkGetSwapchainImagesKHR(device, swapchain, &swapchainImageCount, nullptr);

    std::vector<VkImage> swapchainImages(swapchainImageCount, VK_NULL_HANDLE);
    vkGetSwapchainImagesKHR(device, swapchain, &swapchainImageCount, swapchainImages.data());

    std::vector<VkImageView> swapchainImageViews(swapchainImageCount, VK_NULL_HANDLE);
    for (uint32_t i = 0; i < swapchainImageCount; ++i)
        createImageView(swapchainImageViews[i], swapchainImages[i], VK_IMAGE_VIEW_TYPE_2D, VK_FORMAT_R8G8B8A8_SRGB, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}, device);

    //Create a depth buffer
    VkImageCreateInfo depthImageCreateInfo{};
    depthImageCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    depthImageCreateInfo.imageType = VK_IMAGE_TYPE_2D;
    depthImageCreateInfo.format = VK_FORMAT_D32_SFLOAT;
    depthImageCreateInfo.extent = {static_cast<uint32_t>(width), static_cast<uint32_t>(height), 1};
    depthImageCreateInfo.mipLevels = 1;
    depthImageCreateInfo.arrayLayers = 1;
    depthImageCreateInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    depthImageCreateInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    depthImageCreateInfo.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
    depthImageCreateInfo.initialLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VmaAllocationCreateInfo depthAllocationCreateInfo{};
    depthAllocationCreateInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;
    depthAllocationCreateInfo.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

    VkImage depthImage;
    VmaAllocation depthAllocation;
    if (vmaCreateImage(allocator, &depthImageCreateInfo, &depthAllocationCreateInfo, &depthImage, &depthAllocation, nullptr) != VK_SUCCESS)
        throw std::runtime_error("Failed to create depth image");

    VkImageView depthImageView;
    createImageView(depthImageView, depthImage, VK_IMAGE_VIEW_TYPE_2D, VK_FORMAT_D32_SFLOAT, {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1}, device);

    //Describe and create a render pass
    VkAttachmentDescription colorAttachment{};
    colorAttachment.format = VK_FORMAT_R8G8B8A8_SRGB;
    colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
    colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    colorAttachment.initialLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    VkAttachmentReference colorAttachmentReference{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};

    VkAttachmentDescription depthAttachment{};
    depthAttachment.format = VK_FORMAT_D32_SFLOAT;
    depthAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
    depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depthAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    depthAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depthAttachment.initialLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    depthAttachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkAttachmentReference depthAttachmentReference{1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};

    VkSubpassDescription subpassDescription{};
    subpassDescription.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpassDescription.colorAttachmentCount = 1;
    subpassDescription.pColorAttachments = &colorAttachmentReference;
    subpassDescription.pDepthStencilAttachment = &depthAttachmentReference;

    VkSubpassDependency subpassDependency{};
    subpassDependency.srcSubpass = VK_SUBPASS_EXTERNAL;
    subpassDependency.dstSubpass = 0;
    subpassDependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    subpassDependency.srcAccessMask = 0;
    subpassDependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    subpassDependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

    VkAttachmentDescription renderPassAttachments[] = {colorAttachment, depthAttachment};

    VkRenderPassCreateInfo renderPassCreateInfo{};
    renderPassCreateInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    renderPassCreateInfo.attachmentCount = countof(renderPassAttachments);
    renderPassCreateInfo.pAttachments = renderPassAttachments;
    renderPassCreateInfo.subpassCount = 1;
    renderPassCreateInfo.pSubpasses = &subpassDescription;
    renderPassCreateInfo.dependencyCount = 1;
    renderPassCreateInfo.pDependencies = &subpassDependency;

    VkRenderPass renderPass;
    if (vkCreateRenderPass(device, &renderPassCreateInfo, nullptr, &renderPass) != VK_SUCCESS)
        throw std::runtime_error("Failed to create render pass");

    //Create our main framebuffers
    std::vector<VkFramebuffer> framebuffers(swapchainImageCount);
    for (uint32_t i = 0; i < swapchainImageCount; ++i) {
        VkImageView framebufferAttachments[] = {swapchainImageViews[i], depthImageView};
        VkFramebufferCreateInfo framebufferCreateInfo{};
        framebufferCreateInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        framebufferCreateInfo.renderPass = renderPass;
        framebufferCreateInfo.attachmentCount = countof(framebufferAttachments);
        framebufferCreateInfo.pAttachments = framebufferAttachments;
        framebufferCreateInfo.width = swapchainCreateInfo.imageExtent.width;
        framebufferCreateInfo.height = swapchainCreateInfo.imageExtent.height;
        framebufferCreateInfo.layers = 1;

        if (vkCreateFramebuffer(device, &framebufferCreateInfo, nullptr, &framebuffers[i]) != VK_SUCCESS)
            throw std::runtime_error("Failed to create framebuffer");
    }

    //Read and compile GLSL shaders and create respective shader modules to use in our graphics pipeline
    const auto vertShaderCode = readFile("../vertex.vert");
    const auto fragShaderCode = readFile("../fragment.frag");
    VkShaderModule
    vertexShader = createShaderModule(compileShader(vertShaderCode,shaderc_shader_kind::shaderc_glsl_default_vertex_shader), device),
    fragmentShader = createShaderModule(compileShader(fragShaderCode, shaderc_shader_kind::shaderc_glsl_default_fragment_shader), device);

    //Describe and create a graphics pipeline
    VkPipelineShaderStageCreateInfo vertexStage{};
    vertexStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    vertexStage.stage = VK_SHADER_STAGE_VERTEX_BIT;
    vertexStage.module = vertexShader;
    vertexStage.pName = "main";

    VkPipelineShaderStageCreateInfo fragmentStage{};
    fragmentStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    fragmentStage.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    fragmentStage.module = fragmentShader;
    fragmentStage.pName = "main";

    VkPipelineShaderStageCreateInfo shaderStages[] = {vertexStage, fragmentStage};

    VertexDescription vertexDescription = Vertex::vertexDescription();

    VkPipelineVertexInputStateCreateInfo vertexInputStateCreateInfo{};
    vertexInputStateCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInputStateCreateInfo.vertexBindingDescriptionCount = vertexDescription.bindings.size();
    vertexInputStateCreateInfo.pVertexBindingDescriptions = vertexDescription.bindings.data();
    vertexInputStateCreateInfo.vertexAttributeDescriptionCount = vertexDescription.attributes.size();
    vertexInputStateCreateInfo.pVertexAttributeDescriptions = vertexDescription.attributes.data();

    VkPipelineInputAssemblyStateCreateInfo inputAssemblyStateCreateInfo{};
    inputAssemblyStateCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssemblyStateCreateInfo.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    inputAssemblyStateCreateInfo.primitiveRestartEnable = VK_FALSE;

    VkViewport viewport{};
    viewport.width = (float) swapchainCreateInfo.imageExtent.width;
    viewport.height = (float) swapchainCreateInfo.imageExtent.height;
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;

    VkRect2D scissor{};
    scissor.extent = swapchainCreateInfo.imageExtent;

    VkPipelineViewportStateCreateInfo viewportStateCreateInfo{};
    viewportStateCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportStateCreateInfo.viewportCount = 1;
    viewportStateCreateInfo.pViewports = &viewport;
    viewportStateCreateInfo.scissorCount = 1;
    viewportStateCreateInfo.pScissors = &scissor;

    VkPipelineRasterizationStateCreateInfo rasterizationStateCreateInfo{};
    rasterizationStateCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizationStateCreateInfo.depthClampEnable = VK_FALSE;
    rasterizationStateCreateInfo.depthBiasEnable = VK_FALSE;
    rasterizationStateCreateInfo.rasterizerDiscardEnable = VK_FALSE;
    rasterizationStateCreateInfo.polygonMode = VK_POLYGON_MODE_FILL;
    rasterizationStateCreateInfo.cullMode = VK_CULL_MODE_NONE;
    rasterizationStateCreateInfo.frontFace = VK_FRONT_FACE_CLOCKWISE;
    rasterizationStateCreateInfo.lineWidth = 1.0f;

    VkPipelineMultisampleStateCreateInfo multisampleStateCreateInfo{};
    multisampleStateCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampleStateCreateInfo.sampleShadingEnable = VK_FALSE;
    multisampleStateCreateInfo.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineDepthStencilStateCreateInfo depthStencilStateCreateInfo{};
    depthStencilStateCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depthStencilStateCreateInfo.depthTestEnable = VK_TRUE;
    depthStencilStateCreateInfo.depthWriteEnable = VK_TRUE;
    depthStencilStateCreateInfo.depthCompareOp = VK_COMPARE_OP_LESS;
    depthStencilStateCreateInfo.depthBoundsTestEnable = VK_FALSE;
    depthStencilStateCreateInfo.stencilTestEnable = VK_FALSE;

    VkPipelineColorBlendAttachmentState colorBlendAttachment{};
    colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    colorBlendAttachment.blendEnable = VK_FALSE;

    VkPipelineColorBlendStateCreateInfo colorBlendStateCreateInfo{};
    colorBlendStateCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlendStateCreateInfo.logicOpEnable = VK_FALSE;
    colorBlendStateCreateInfo.logicOp = VK_LOGIC_OP_COPY;
    colorBlendStateCreateInfo.attachmentCount = 1;
    colorBlendStateCreateInfo.pAttachments = &colorBlendAttachment;

    VkGraphicsPipelineCreateInfo graphicsPipelineCreateInfo{};
    graphicsPipelineCreateInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    graphicsPipelineCreateInfo.stageCount = countof(shaderStages);
    graphicsPipelineCreateInfo.pStages = shaderStages;
    graphicsPipelineCreateInfo.pVertexInputState = &vertexInputStateCreateInfo;
    graphicsPipelineCreateInfo.pInputAssemblyState = &inputAssemblyStateCreateInfo;
    graphicsPipelineCreateInfo.pViewportState = &viewportStateCreateInfo;
    graphicsPipelineCreateInfo.pRasterizationState = &rasterizationStateCreateInfo;
    graphicsPipelineCreateInfo.pMultisampleState = &multisampleStateCreateInfo;
    graphicsPipelineCreateInfo.pDepthStencilState = &depthStencilStateCreateInfo;
    graphicsPipelineCreateInfo.pColorBlendState = &colorBlendStateCreateInfo;
    graphicsPipelineCreateInfo.renderPass = renderPass;
    graphicsPipelineCreateInfo.subpass = 0;

    //TODO: Automate this using SPIRV-Reflect
    //Describe the layout of our pipeline structures and create a pipeline layout
    VkPushConstantRange pushConstantRange{};
    pushConstantRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    pushConstantRange.offset = 0;
    pushConstantRange.size = sizeof(glm::mat4);

    VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo{};
    pipelineLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutCreateInfo.setLayoutCount = 0;
    pipelineLayoutCreateInfo.pSetLayouts = nullptr;
    pipelineLayoutCreateInfo.pushConstantRangeCount = 1;
    pipelineLayoutCreateInfo.pPushConstantRanges = &pushConstantRange;

    VkPipelineLayout pipelineLayout;
    if (vkCreatePipelineLayout(device, &pipelineLayoutCreateInfo, nullptr, &pipelineLayout) != VK_SUCCESS)
        throw std::runtime_error("Failed to create main pipeline layout");

    graphicsPipelineCreateInfo.layout = pipelineLayout;

    VkPipeline graphicsPipeline;
    if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &graphicsPipelineCreateInfo, nullptr, &graphicsPipeline) != VK_SUCCESS)
        throw std::runtime_error("Failed to create main graphics pipeline");

    //Create a command pool to contain drawing command buffers
    VkCommandPoolCreateInfo commandPoolCreateInfo{};
    commandPoolCreateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    commandPoolCreateInfo.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT | VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    commandPoolCreateInfo.queueFamilyIndex = graphicsFamily;

    VkCommandPool commandPool;
    if (vkCreateCommandPool(device, &commandPoolCreateInfo, nullptr, &commandPool) != VK_SUCCESS)
        throw std::runtime_error("Failed to create main draw command pool");

    //Allocate a command buffer for each swapchain image to asynchronously submit
    VkCommandBufferAllocateInfo commandBufferAllocateInfo{};
    commandBufferAllocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    commandBufferAllocateInfo.commandPool = commandPool;
    commandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    commandBufferAllocateInfo.commandBufferCount = swapchainImageCount;

    std::vector<VkCommandBuffer> commandBuffers(swapchainImageCount, VK_NULL_HANDLE);
    if (vkAllocateCommandBuffers(device, &commandBufferAllocateInfo, commandBuffers.data()) != VK_SUCCESS)
        throw std::runtime_error("Failed to allocate main draw command buffers");

    //Setup some synchronization objects
    VkSemaphoreCreateInfo semaphoreCreateInfo{};
    semaphoreCreateInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

    VkFenceCreateInfo fenceCreateInfo{};
    fenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceCreateInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    std::vector<VkSemaphore> swapchainLockSemaphore(swapchainImageCount, VK_NULL_HANDLE);
    std::vector<VkSemaphore> renderLockSemaphore(swapchainImageCount, VK_NULL_HANDLE);
    std::vector<VkFence> frameLockFence(swapchainImageCount, VK_NULL_HANDLE);

    for (uint32_t i = 0; i < swapchainImageCount; ++i) {
        vkCreateSemaphore(device, &semaphoreCreateInfo, nullptr, &swapchainLockSemaphore[i]);
        vkCreateSemaphore(device, &semaphoreCreateInfo, nullptr, &renderLockSemaphore[i]);
        vkCreateFence(device, &fenceCreateInfo, nullptr, &frameLockFence[i]);
    }

    //Load sample mesh
    Mesh mesh = Mesh::loadObj("../dragon.obj");

    //Create a simple vertex & index buffer
    VkBufferCreateInfo bufferCreateInfo{};
    bufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;

    VmaAllocationCreateInfo allocationCreateInfo{};
    allocationCreateInfo.usage = VMA_MEMORY_USAGE_CPU_TO_GPU;

    bufferCreateInfo.size = static_cast<VkDeviceSize>(mesh.vertices.size()*sizeof(Vertex));
    bufferCreateInfo.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;

    VkBuffer vertexBuffer;
    VmaAllocation vertexAllocation;
    if (vmaCreateBuffer(allocator, &bufferCreateInfo, &allocationCreateInfo, &vertexBuffer, &vertexAllocation, nullptr) != VK_SUCCESS)
        throw std::runtime_error("Failed to allocate vertex buffer");

    bufferCreateInfo.size = static_cast<VkDeviceSize>(mesh.indices.size()*sizeof(uint32_t));
    bufferCreateInfo.usage = VK_BUFFER_USAGE_INDEX_BUFFER_BIT;

    VkBuffer indexBuffer;
    VmaAllocation indexAllocation;
    if (vmaCreateBuffer(allocator, &bufferCreateInfo, &allocationCreateInfo, &indexBuffer, &indexAllocation, nullptr) != VK_SUCCESS)
        throw std::runtime_error("Failed to allocate index buffer");

    //Fill the vertex & index buffers
    void* data;

    vmaMapMemory(allocator, vertexAllocation, &data);
    std::memcpy(data, mesh.vertices.data(), mesh.vertices.size()*sizeof(Vertex));
    vmaUnmapMemory(allocator, vertexAllocation);

    vmaMapMemory(allocator, indexAllocation, &data);
    std::memcpy(data, mesh.indices.data(), mesh.indices.size()*sizeof(uint32_t));
    vmaUnmapMemory(allocator, indexAllocation);

    //Main rendering loop
    uint32_t frame = 0;
    const auto start = std::chrono::high_resolution_clock::now(); //Timestamp for rendering start
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        if (glfwGetWindowAttrib(window, GLFW_ICONIFIED) == GLFW_TRUE) continue;

        vkWaitForFences(device, 1, &frameLockFence[frame], VK_TRUE, UINT64_MAX); //Wait for this frame to be unlocked (CPU)
        vkResetFences(device, 1, &frameLockFence[frame]); //Re-lock this frame (CPU)

        //Get the index of the next available image in the swapchain and assign that image to this frame, not guaranteed to be in order
        //The acquire function does not block this thread, so instead we make it signal a semaphore when the function returns an available image, so we can synchronize our queues for this frame
        uint32_t swapchainIndex = 0;
        vkAcquireNextImageKHR(device, swapchain, UINT64_MAX, swapchainLockSemaphore[frame], VK_NULL_HANDLE, &swapchainIndex);

        VkCommandBufferBeginInfo commandBufferBeginInfo{};
        commandBufferBeginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        commandBufferBeginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

        //Implicit command buffer reset
        if (vkBeginCommandBuffer(commandBuffers[frame], &commandBufferBeginInfo) != VK_SUCCESS)
            throw std::runtime_error("Failed to begin recording command buffer");

        VkClearValue clearValues[2];
        clearValues[0].color = {{0.0f, 0.0f, 0.0f, 0.0f}};
        clearValues[1].depthStencil = {1.0f, 0};

        VkRenderPassBeginInfo renderPassBeginInfo{};
        renderPassBeginInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        renderPassBeginInfo.renderPass = renderPass;
        renderPassBeginInfo.framebuffer = framebuffers[frame];
        renderPassBeginInfo.renderArea = scissor;
        renderPassBeginInfo.clearValueCount = countof(clearValues);
        renderPassBeginInfo.pClearValues = clearValues;

        vkCmdBeginRenderPass(commandBuffers[frame], &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);

        vkCmdBindPipeline(commandBuffers[frame], VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);

        VkDeviceSize offset = 0;
        vkCmdBindVertexBuffers(commandBuffers[frame], 0, 1, &vertexBuffer, &offset);
        vkCmdBindIndexBuffer(commandBuffers[frame], indexBuffer, offset, VK_INDEX_TYPE_UINT32);

        float time = std::chrono::duration_cast<std::chrono::duration<float, std::ratio<1,1>>>(std::chrono::high_resolution_clock::now()-start).count();

        glm::mat4 model = glm::scale(glm::vec3(1,1,1))*glm::rotate(glm::radians(time*90.0f), glm::vec3(0.0f, 1.0f, 0.0f))*glm::translate(glm::vec3(0,0,0));
        glm::mat4 view = glm::lookAt(glm::vec3(0.0f, 0.5f, -1.0f), glm::vec3(0,0,0), glm::vec3(0,1,0));
        glm::mat4 projection = glm::perspective(glm::radians(70.0f), (float)width/(float)height, 0.01f, 100.0f);
        projection[1][1] *= -1;

        glm::mat4 renderMatrix = projection*view*model;

        vkCmdPushConstants(commandBuffers[frame], pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(renderMatrix), &renderMatrix);

        vkCmdDrawIndexed(commandBuffers[frame], mesh.indices.size(), 1, 0, 0, 0);

        vkCmdEndRenderPass(commandBuffers[frame]);

        if (vkEndCommandBuffer(commandBuffers[frame]) != VK_SUCCESS)
            throw std::runtime_error("Failed to record command buffer");

        VkPipelineStageFlags waitStages = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffers[frame];
        submitInfo.pWaitDstStageMask = &waitStages;
        submitInfo.waitSemaphoreCount = 1;
        submitInfo.pWaitSemaphores = &swapchainLockSemaphore[frame]; //Don't start until new image available (GPU)
        submitInfo.signalSemaphoreCount = 1;
        submitInfo.pSignalSemaphores = &renderLockSemaphore[frame]; //Unlock render lock when rendering finished (GPU)

        //Unlock frame lock after rendering on that frame is finished (CPU)
        if (vkQueueSubmit(graphicsQueue, 1, &submitInfo, frameLockFence[frame]) != VK_SUCCESS)
            throw std::runtime_error("Couldn't render ;(");

        VkPresentInfoKHR presentInfo{};
        presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
        presentInfo.swapchainCount = 1;
        presentInfo.pSwapchains = &swapchain;
        presentInfo.pImageIndices = &swapchainIndex;
        presentInfo.waitSemaphoreCount = 1;
        presentInfo.pWaitSemaphores = &renderLockSemaphore[frame]; //Wait until image is rendered (GPU)

        if (vkQueuePresentKHR(graphicsQueue, &presentInfo) != VK_SUCCESS)
            throw std::runtime_error("Couldn't present rendered image, how did you fuck up?");

        frame = (frame + 1) % swapchainImageCount; //Jump to next frame
    }

    //Destroy sample mesh components
    vmaDestroyBuffer(allocator, indexBuffer, indexAllocation);
    vmaDestroyBuffer(allocator, vertexBuffer, vertexAllocation);

    //On exit, wait until all operations on device are finished
    vkDeviceWaitIdle(device);

    //Destroy all structures in order
    for (uint32_t i = 0; i < swapchainImageCount; ++i) {
        vkDestroySemaphore(device, swapchainLockSemaphore[i], nullptr);
        vkDestroySemaphore(device, renderLockSemaphore[i], nullptr);
        vkDestroyFence(device, frameLockFence[i], nullptr);
    }
    vkFreeCommandBuffers(device, commandPool, swapchainImageCount, commandBuffers.data());
    vkDestroyCommandPool(device, commandPool, nullptr);
    vkDestroyShaderModule(device, vertexShader, nullptr);
    vkDestroyShaderModule(device, fragmentShader, nullptr);
    vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
    vkDestroyPipeline(device, graphicsPipeline, nullptr);
    for (uint32_t i = 0; i < swapchainImageCount; ++i)
        vkDestroyFramebuffer(device, framebuffers[i], nullptr);
    vkDestroyRenderPass(device, renderPass, nullptr);
    vkDestroyImageView(device, depthImageView, nullptr);
    vmaDestroyImage(allocator, depthImage, depthAllocation);
    for (uint32_t i = 0; i < swapchainImageCount; ++i)
        vkDestroyImageView(device, swapchainImageViews[i], nullptr);
    vkDestroySwapchainKHR(device, swapchain, nullptr);
    vkDestroySurfaceKHR(instance, surface, nullptr);
    vmaDestroyAllocator(allocator);
    vkDestroyDevice(device, nullptr);
    vkDestroyInstance(instance, nullptr);

    glfwTerminate();
    return 0;
}

//TODO Properly implement an object tracking system that auto destroys unneeded Vulkan objects
//TODO Make a fail function that safely terminates the program by manually destroying all objects before crash