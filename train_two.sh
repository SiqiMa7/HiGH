#python two_branch/mhgcn.py --dataset movielen_rel_two --data_path datasets/movielen_rel.pk --loss-lambda 0.8 \
#--num-hidden 8 --num-heads 8 --spm --residual --norm --loss-alpha 0.3 --list-num 100 \
#--save-path gtran-list-3-100_checkpoint.pt --gpu 0

python two_branch/mhgcn_douban_all.py --dataset doubanmovie_rel_two --data_path datasets/doubanmovie_short_rel.pk --loss-lambda 0.8 \
--num-hidden 8 --num-heads 8 --spm --residual --norm --loss-alpha 0.3 --list-num 100 \
--save-path gtran-list-3-100_checkpoint.pt --gpu 1

# python two_branch/mhgcn_DB_all.py --dataset doubanbook_rel_two --data_path datasets/doubanbook_rel.pk --loss-lambda 0.8 \
# --num-hidden 8 --num-heads 8 --spm --residual --norm --loss-alpha 0.3 --list-num 100 \
# --save-path gtran-list-3-100_checkpoint.pt --gpu 0


