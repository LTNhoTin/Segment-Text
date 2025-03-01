from wtpsplit import SaT
from multiprocessing import freeze_support
import torch
from utils import device

if __name__ == "__main__":
    freeze_support()
    model = SaT("sat-3l")
    device = device()
    model.to(device)

    # Test segmentation
    text = "Giá vàng giảm hơn 2% do nhà đầu tư chốt lời sau khi kim loại quý này lập đỉnh mới hôm 24/2. Trưa 25/2 (rạng sáng 26/2 giờ Hà Nội), giá vàng giao ngay bốc hơi 60 USD, xuống còn 2.893 USD một ouce. Mức này giảm khoảng 2% so với chốt phiên giao dịch hôm trước. David Meger, Giám đốc giao dịch kim loại tại High Ridge Futures lý giải đợt lao dốc này là hoạt động chốt lời thông thường sau khi vàng lập đỉnh mới. Trước đó, vàng thế giới giao ngay tăng 18 USD lên 2.953 USD một ounce trong phiên giao dịch hôm 24/2. Kim loại quý có lúc xác lập kỷ lục mới tại 2.956 USD. Đây là lần thứ 11 kim loại quý lập đỉnh trong năm nay."
    print(model.split(text,block_size=4,stride=4))
