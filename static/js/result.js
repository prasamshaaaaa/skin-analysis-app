document.addEventListener('DOMContentLoaded', () => {
    const showBtn = document.getElementById('showProducts');
    const productsDiv = document.getElementById('products');

    showBtn.addEventListener('click', () => {
        if (productsDiv.style.display === "none" || productsDiv.style.display === "") {
            productsDiv.style.display = "grid";
            productsDiv.classList.add('animate__animated', 'animate__fadeIn');
            showBtn.textContent = "Hide Recommended Products";
        } else {
            productsDiv.style.display = "none";
            showBtn.textContent = "Show Recommended Products";
        }
    });
});
