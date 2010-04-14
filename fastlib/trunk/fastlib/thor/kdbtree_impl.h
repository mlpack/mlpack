/* Template implementations for kdtree.h. */

template<typename TPoint, typename TNode, typename TParam>
void KdBTreeBuilder<TPoint, TNode, TParam>::Doit(
    fx_module* module, const Param* param_in,
    index_t begin_index, index_t end_index,
    DistributedCache* input_in,
    DistributedCache* points_create,
    DistributedCache* nodes_create,
    TreeDecomposition* decomposition) {
  DEBUG_ASSERT_MSG(rpc::n_peers() == 1, "kdB-trees only coded for serial!");

  param_ = param_in;
  n_points_ = end_index - begin_index;

  inputs_.Init(input_in, BlockDevice::M_READ);
  points_.Init(points_create, BlockDevice::M_CREATE);
  nodes_.Init(nodes_create, BlockDevice::M_CREATE);

  index_t dimension;
  {
    CacheRead<Point> first_point(&inputs_, inputs_.begin_index());
    dimension = first_point->vec().length();
  }

  point_block_size_ = points_.n_block_elems();
  node_block_size_ = nodes_.n_block_elems();

  leaf_size_ = fx_param_int(module, "leaf_size", 32);
  if (leaf_size_ > point_block_size_) {
    NONFATAL("Decreasing leaf size from %d to %d due to block size!\n",
       int(leaf_size_), int(point_block_size_));
    leaf_size_ = point_block_size_;
  }

  spare_block_i_ = -index_t(1);

  fx_timer_start(module, "tree_build");

  index_t node_i = Build_(begin_index, end_index);
  decomposition->Init(new DecompNode(
        typename TreeDecomposition::Info(0, 1),
        &nodes_, node_i, nodes_.end_index()));

  fx_timer_stop(module, "tree_build");
}

template<typename TPoint, typename TNode, typename TParam>
index_t KdBTreeBuilder<TPoint, TNode, TParam>::Build_(
    index_t begin_col, index_t end_col) {
  index_t point_i = points_.AllocD(0, point_block_size_);
  index_t root_i = nodes_.AllocD(0, node_block_size_);
  {
    CacheWrite<Node> root(&nodes_, root_i);
    root->set_range(point_i, 0);
    root->bound().Reset();
    root->set_leaf();
    root->set_parent(-index_t(1));
    root->set_subnodes_in_page(1);
  }

  NOTIFY("Phase 1 of tree building...");

  /* Add points one by one */
  CacheReadIter<Point> input(&inputs_, begin_col);
  for (int i = end_col - begin_col; i--; input.Next()) {
    Insert_(*input, root_i);
    fl_print_progress("points in tree",
        (n_points_ - i) * 100 / n_points_);
  }

  NOTIFY("Phase 2 of tree building...");

  /* Build leaf-level partitions */
  SplitLeafPages_(root_i);

  NOTIFY("Phase 3 of tree building...");

  /* Fill statistics and node ranges */
  CacheWrite<Node> root(&nodes_, root_i);
  Postprocess_(&*root);

  NOTIFY("Done!");

  return root_i;
}

template<typename TPoint, typename TNode, typename TParam>
void KdBTreeBuilder<TPoint, TNode, TParam>::Postprocess_(
    Node *node) {
  if (!node->is_leaf()) {
    /* Recurse on children */
    CacheWrite<Node> left(&nodes_, node->child(0));
    CacheWrite<Node> right(&nodes_, node->child(1));
    Postprocess_(&*left);
    Postprocess_(&*right);

    /* Update ranges; done here due to shuffling */
    node->set_range(min(left->begin(), right->begin()),
		    max(left->end(), right->end()),
		    left->count() + right->count());

    /* Update stats; done here to reduce cost */
    node->stat().Accumulate(*param_, left->stat(),
			    left->bound(), left->count());
    node->stat().Accumulate(*param_, right->stat(),
			    right->bound(), right->count());
  } else {
    /* Finally compute stats for the leaf */
    CacheReadIter<Point> point_iter(&points_, node->begin());
    for (index_t remaining = node->count(); remaining--;
	 point_iter.Next()) {
      node->stat().Accumulate(*param_, *point_iter);
    }
  }

  /* Both leafs and inner node stats need postprocessing */
  node->stat().Postprocess(*param_, node->bound(), node->count());
}

template<typename TPoint, typename TNode, typename TParam>
index_t KdBTreeBuilder<TPoint, TNode, TParam>::SplitLeafPages_(
    index_t node_i) {
  const Node *node = nodes_.StartRead(node_i);

  if (!node->is_leaf()) {
    /* Recurse on children */
    for (int k = 0; k < Node::CARDINALITY; k++) {
      index_t child_i = node->child(k);
      nodes_.StopRead(node_i);
      node_i = SplitLeafPages_(child_i);
      node = nodes_.StartRead(node_i);
    }
  } else if (node->count() > leaf_size_) {
    /* Fits in block, but split to cut cost of base-case */
    nodes_.StopRead(node_i);
    node_i = SplitLeaf_(node_i, false);
    return SplitLeafPages_(node_i);
  }

  /* Parent may have moved due to splitting! */
  index_t parent_i = node->parent();
  nodes_.StopRead(node_i);
  return parent_i;
}

template<typename TPoint, typename TNode, typename TParam>
void KdBTreeBuilder<TPoint, TNode, TParam>::Insert_(
    const Point &input, index_t node_i) {
  /* Find the leaf to which the new point belongs */
  Node *node = nodes_.StartWrite(node_i);
  for (;;) {
    /* Incorporate the point into nodes' bounding boxes */
    node->bound() |= input.vec();

    if (node->is_leaf()) {
      break; /* The point will go here! */
    }

    index_t left_i = node->child(0);
    index_t right_i = node->child(1);
    nodes_.StopWrite(node_i);

    /* Decide if we move into the left or right child */
    Node *left = nodes_.StartWrite(left_i);
    Node *right = nodes_.StartWrite(right_i);
    if (left->bound().MinDistanceSq(input.vec())
	<= right->bound().MinDistanceSq(input.vec())) {
      node_i = left_i;
      node = left;
      nodes_.StopWrite(right_i);
    } else {
      node_i = right_i;
      node = right;
      nodes_.StopWrite(left_i);
    }
  }

  /* Do we need to split? */
  if (node->count() >= point_block_size_) {
    nodes_.StopWrite(node_i);
    node_i = SplitLeaf_(node_i, true);
    Insert_(input, node_i);
  } else {
    CacheWrite<Point> point(&points_, node->begin() + node->count());
    mem::CopyBytes(&*point, &input, points_.n_elem_bytes());
    ot::SemiCopy<Point>(reinterpret_cast<char *>(&*point), &input);

    node->set_range(node->begin(), node->count() + 1);
    nodes_.StopWrite(node_i);
  }
}

template<typename TPoint, typename TNode, typename TParam>
index_t KdBTreeBuilder<TPoint, TNode, TParam>::SplitLeaf_(
    index_t node_i, bool make_new_block) {
  /* Add children to the newly split leaf; may restructure tree */
  node_i = CreateChildren_(node_i);
  CacheRead<Node> node(&nodes_, node_i);
  CacheWrite<Node> left(&nodes_, node->child(0));
  CacheWrite<Node> right(&nodes_, node->child(1));

  left->bound().Reset();
  left->set_leaf();
  left->set_parent(node_i);
  left->set_subnodes_in_page(0);

  right->bound().Reset();
  right->set_leaf();
  right->set_parent(node_i);
  right->set_subnodes_in_page(0);

  MedianSplit_(&*node, &left->bound(), &right->bound());

  /* Copy right side's points into new block, if appropriate */
  index_t point_i = node->begin() + node->count() / 2;
  if (make_new_block) {
    CacheReadIter<Point> old_iter(&points_, point_i);
    point_i = points_.AllocD(0, point_block_size_);
    CacheWriteIter<Point> new_iter(&points_, point_i);
    for (int remaining = node->count() / 2; remaining--;
	 old_iter.Next(), new_iter.Next()) {
      mem::CopyBytes(&*new_iter, &*old_iter, points_.n_elem_bytes());
      ot::SemiCopy(reinterpret_cast<char *>(&*new_iter), &*old_iter);
    }
  }

  /* Tell children where their points are */
  left->set_range(node->begin(), node->count() / 2);
  right->set_range(point_i, node->count() / 2);

  return node_i;
}

template<typename TPoint, typename TNode, typename TParam>
index_t KdBTreeBuilder<TPoint, TNode, TParam>::CreateChildren_(
    index_t node_i) {
  /* Find the parent of the current tree segment */
  index_t parent_i = node_i;
  Node *parent;
  for (;;) {
    /* We're adding two subnodes along a leaf-to-root path */
    parent = nodes_.StartWrite(parent_i);
    parent->set_subnodes_in_page(parent->subnodes_in_page() + 2);

    if (parent->parent() == -index_t(1)
	|| nodes_.Blockid(parent_i) != nodes_.Blockid(node_i)) {
      /* The parent is the first node outside the current block */
      break;
    }

    /* Move up a node; note: added "parent()" to thor_struct.h */
    index_t next_i = parent->parent();
    nodes_.StopWrite(parent_i);
    parent_i = next_i;
  }

  /* If there's room, add children; note: left child begins page */
  if (parent->subnodes_in_page() <= node_block_size_) {
    index_t block_i = parent->parent() == -index_t(1)
      ? parent_i : parent->child(0);
    CacheWrite<Node> node(&nodes_, node_i);
    node->set_child(0, block_i + parent->subnodes_in_page() - 2);
    node->set_child(1, block_i + parent->subnodes_in_page() - 1);
    nodes_.StopWrite(parent_i);

    return node_i;
  }

  /* Special handling for root page */
  if (parent->parent() == -index_t(1)) {
    index_t block_i = spare_block_i_ != -index_t(1)
      ? spare_block_i_
      : nodes_.AllocD(0, node_block_size_);

    index_t left_i = parent->child(0);
    index_t right_i = parent->child(1);
    CacheWrite<Node> left(&nodes_, left_i);
    CacheWrite<Node> right(&nodes_, right_i);

    if (left->is_leaf() ||
	nodes_.Blockid(left->child(0)) != nodes_.Blockid(left_i)) {
      if (node_i == left_i) {
	left->set_child(0, block_i);
	left->set_child(1, block_i + 1);
	block_i = nodes_.AllocD(0, node_block_size_);
      }
      PackNodes_(block_i, right_i, &*right, 0, &node_i);
    } else if (right->is_leaf() ||
	nodes_.Blockid(right->child(0)) != nodes_.Blockid(right_i)) {
      if (node_i == right_i) {
	right->set_child(0, block_i);
	right->set_child(1, block_i + 1);
	block_i = nodes_.AllocD(0, node_block_size_);
      }
      PackNodes_(block_i, left_i, &*left, 0, &node_i);
    } else {
      PackNodes_(block_i, left_i, &*left, 0, &node_i);
      block_i = nodes_.AllocD(0, node_block_size_);
      PackNodes_(block_i, right_i, &*right, 0, &node_i);
    }

    parent->set_subnodes_in_page(3);
    nodes_.StopWrite(parent_i);

    spare_block_i_ = -index_t(1);
    return node_i;
  }

  /* Remember old children; note: old_left_i is first in block */
  index_t old_left_i = parent->child(0);
  index_t old_right_i = parent->child(1);

  /* Children should be in the same, different block from parent */
  DEBUG_ASSERT((old_left_i & nodes_.n_block_elems_mask()) == 0);
  DEBUG_ASSERT(nodes_.Blockid(old_left_i) == nodes_.Blockid(old_right_i));
  DEBUG_ASSERT(nodes_.Blockid(old_left_i) != nodes_.Blockid(parent_i));

  /* Old page parent isn't one any more, so reset subnode count */
  parent->set_subnodes_in_page(0);
  nodes_.StopWrite(parent_i);
  parent_i = CreateChildren_(parent_i);

  /* Identify new children */
  parent = nodes_.StartWrite(parent_i);
  index_t new_left_i = parent->child(0);
  index_t new_right_i = parent->child(1);
  nodes_.StopWrite(parent_i);

  /* New page for split, or at least for work */
  index_t block_i = spare_block_i_ != -index_t(1)
    ? spare_block_i_
    : nodes_.AllocD(0, node_block_size_);
  index_t for_copy_i = block_i;

  /* Copy left child up, fixing parent ref */
  CacheWrite<Node> new_left(&nodes_, new_left_i);
  CacheRead<Node> old_left(&nodes_, old_left_i);
  mem::CopyBytes(&*new_left, &*old_left, nodes_.n_elem_bytes());
  ot::SemiCopy(reinterpret_cast<char *>(&*new_left), &*old_left);
  new_left->set_parent(parent_i);

  /* Copy right child up, fixing parent ref */
  CacheWrite<Node> new_right(&nodes_, new_right_i);
  CacheRead<Node> old_right(&nodes_, old_right_i);
  mem::CopyBytes(&*new_right, &*old_right, nodes_.n_elem_bytes());
  ot::SemiCopy(reinterpret_cast<char *>(&*new_right), &*old_right);
  new_right->set_parent(parent_i);

  /* Special handling if copied node is a stub */
  if (new_left->is_leaf() ||
      nodes_.Blockid(new_left->child(0)) != nodes_.Blockid(old_left_i)) {
    if (node_i == old_left_i) {
      /* Worst-case, essentially; we still need the block */
      node_i = new_left_i;
      new_left->set_child(0, block_i);
      new_left->set_child(1, block_i + 1);
      block_i += 2;
      for_copy_i = block_i;
    } else {
      spare_block_i_ = block_i;
      if (!new_left->is_leaf()) {
	/* Still fix refs for nodes pointing to another block */
	FixChildrenParents_(new_left_i, &*new_left);
      }
    }

    /* Pack into new block, to be copied back (hence offsets) */
    block_i = PackNodes_(block_i, new_right_i, &*new_right,
			 old_left_i - block_i, &node_i);
    DEBUG_ASSERT(block_i - for_copy_i == new_right->subnodes_in_page());
  } else if (new_right->is_leaf() ||
      nodes_.Blockid(new_right->child(0)) != nodes_.Blockid(old_left_i)) {
    if (node_i == old_right_i) {
      /* Worst-case, essentially; we still need the block */
      node_i = new_right_i;
      new_right->set_child(0, block_i);
      new_right->set_child(1, block_i + 1);
      block_i += 2;
      for_copy_i = block_i;
    } else {
      spare_block_i_ = block_i;
      if (!new_right->is_leaf()) {
	/* Still fix refs for nodes pointing to another block */
	FixChildrenParents_(new_right_i, &*new_right);
      }
    }

    /* Pack into new block, to be copied back (hence offsets) */
    block_i = PackNodes_(block_i, new_left_i, &*new_left,
			   old_left_i - block_i, &node_i);
    DEBUG_ASSERT(block_i - for_copy_i == new_left->subnodes_in_page());
  } else {
    spare_block_i_ = -index_t(1);

    /* Pack right side into the new block */
    block_i = PackNodes_(block_i, new_right_i, &*new_right, 0, &node_i);
    DEBUG_ASSERT(block_i - for_copy_i == new_right->subnodes_in_page());

    for_copy_i = block_i;

    /* Pack left side in prep to copy it back */
    block_i = PackNodes_(block_i, new_left_i, &*new_left,
			 old_left_i - block_i, &node_i);
    DEBUG_ASSERT(block_i - for_copy_i == new_left->subnodes_in_page());
  }

  /* Copy back to orig (refs already fixed) */
  CacheWriteIter<Node> old_iter(&nodes_, old_left_i);
  CacheReadIter<Node> new_iter(&nodes_, for_copy_i);
  for (int remaining = block_i - for_copy_i; remaining--;
       old_iter.Next(), new_iter.Next()) {
    mem::CopyBytes(&*old_iter, &*new_iter, nodes_.n_elem_bytes());
    ot::SemiCopy(reinterpret_cast<char *>(&*old_iter), &*new_iter);
  }

  if (node_i >= for_copy_i) {
    node_i += old_left_i - for_copy_i;
  }

  return node_i;
}

template<typename TPoint, typename TNode, typename TParam>
index_t KdBTreeBuilder<TPoint, TNode, TParam>::PackNodes_(
    index_t dest_i, index_t parent_i, Node *parent,
    index_t offset_i, index_t *node_ip) {
  for (int k = 0; k < Node::CARDINALITY; ++k) {
    index_t child_i = parent->child(k);
    index_t next_i = dest_i + offset_i;
    parent->set_child(k, next_i);

    CacheWrite<Node> dest(&nodes_, dest_i);
    CacheRead<Node> child(&nodes_, child_i);
    mem::CopyBytes(&*dest, &*child, nodes_.n_elem_bytes());
    ot::SemiCopy(reinterpret_cast<char *>(&*dest), &*child);
    dest->set_parent(parent_i);

    ++dest_i;

    if (node_ip && child_i == *node_ip) {
      /* Found splitting node; leave room for children. */
      *node_ip = next_i - offset_i;
      dest->set_child(0, next_i + 1);
      dest->set_child(1, next_i + 2);
      dest_i += 2;
    } else if (!dest->is_leaf()) {
      if (nodes_.Blockid(dest->child(0)) == nodes_.Blockid(child_i)) {
	/* Children in same block; recurse */
	dest_i = PackNodes_(dest_i, next_i, &*dest, offset_i, node_ip);
      } else {
	/* Still fix parent refs for out-of-block */
	FixChildrenParents_(next_i, &*dest);
      }
    }
  }

  return dest_i;
}

template<typename TPoint, typename TNode, typename TParam>
void KdBTreeBuilder<TPoint, TNode, TParam>::FixChildrenParents_(
    index_t parent_i, const Node *parent) {
  DEBUG_ASSERT(!parent->is_leaf());
  CacheWrite<Node> left(&nodes_, parent->child(0));
  CacheWrite<Node> right(&nodes_, parent->child(1));
  left->set_parent(parent_i);
  right->set_parent(parent_i);
}

template<typename TPoint, typename TNode, typename TParam>
void KdBTreeBuilder<TPoint, TNode, TParam>::MedianSplit_(
    const Node *node, Bound *left_bound, Bound *right_bound) {
  /* Short loop to find widest dimension */
  index_t split_dim = BIG_BAD_NUMBER;
  double max_width = -1;
  for (index_t d = 0; d < node->bound().dim(); d++) {
    double w = node->bound().get(d).width();

    if (w > max_width) {
      max_width = w;
      split_dim = d;
    }
  }

  index_t goal_pos = node->begin() + node->count() / 2;
  index_t begin_pos = node->begin();
  index_t end_pos = node->end();
  DRange current_range = node->bound().get(split_dim);

  Bound temp_left_bound;
  Bound temp_right_bound;
  temp_left_bound.Init(node->bound().dim());
  temp_right_bound.Init(node->bound().dim());

  index_t split_pos;
  for (;;) {
    /* Choose a split val likely close to median */
    double split_val = current_range.interpolate(
        (goal_pos - begin_pos) / double(end_pos - begin_pos));

    /* Enact the split, rearranging points */
    temp_left_bound.Reset();
    temp_right_bound.Reset();
    split_pos = thor::Partition(
        HrectPartitionCondition(split_dim, split_val),
	begin_pos, end_pos - begin_pos, &points_,
	&temp_left_bound, &temp_right_bound);

    /* Pull in range to more tightly bound median */
    if (split_pos == goal_pos) {
      *left_bound |= temp_left_bound;
      *right_bound |= temp_right_bound;
      break;
    } else if (split_pos < goal_pos) {
      *left_bound |= temp_left_bound;
      current_range = temp_right_bound.get(split_dim);
      if (current_range.width() == 0) {
	break; /* Identical elements */
      }
      begin_pos = split_pos;
    } else if (split_pos > goal_pos) {
      *right_bound |= temp_right_bound;
      current_range = temp_left_bound.get(split_dim);
      if (current_range.width() == 0) {
	break; /* Identical elements */
      }
      end_pos = split_pos;
    }    
  }

  if (split_pos != goal_pos) {
    /* Identical elements; compute actual bound */
    FindBoundingBox_(begin_pos, goal_pos, left_bound);
    FindBoundingBox_(goal_pos, end_pos, right_bound);
  }
}

template<typename TPoint, typename TNode, typename TParam>
void KdBTreeBuilder<TPoint, TNode, TParam>::FindBoundingBox_(
    index_t begin_index, index_t end_index, Bound* bound) {
  CacheReadIter<Point> point(&points_, begin_index);
  for (index_t i = end_index - begin_index; i--; point.Next()) {
    *bound |= point->vec();
  }
}

template<typename Point, typename Node, typename Param>
void thor::CreateKdBTreeMaster(const Param& param,
    int points_channel, int nodes_channel,
    int block_size_kb, double megs, datanode *module,
    index_t n_points, DistributedCache *input_cache,
    DistributedCache *points_cache, DistributedCache *nodes_cache,
    ThorTreeDecomposition<Node> *decomposition) {
  Point example_point;
  CacheArray<Point>::GetDefaultElement(input_cache, &example_point);

  CacheArray<Point>::CreateCacheMaster(points_channel,
      CacheArray<Point>::ConvertBlockSize(example_point, block_size_kb),
      example_point, megs, points_cache);

  Node example_node;
  example_node.stat().Init(param);
  example_node.bound().Init(example_point.vec().length());

  CacheArray<Node>::CreateCacheMaster(nodes_channel,
      CacheArray<Node>::ConvertBlockSize(example_node, block_size_kb),
      example_node, megs, nodes_cache);

  KdBTreeBuilder<Point, Node, Param> builder;
  builder.Doit(module, &param, 0, n_points, input_cache,
	       points_cache, nodes_cache, decomposition);
}

template<typename Point, typename Node, typename Param>
void thor::CreateKdBTree(const Param& param,
    int points_channel, int nodes_channel, int extra_channel,
    datanode *module, index_t n_points, DistributedCache *input_cache,
    ThorTree<Param, Point, Node> *tree_out) {
  DistributedCache *points_cache = new DistributedCache();
  DistributedCache *nodes_cache = new DistributedCache();

  double megs = fx_param_double(module, "megs", 1000);
  int block_size_kb = fx_param_int(module, "block_size_kb", 64);

  /* Not sure what all this does; might be important */
  Broadcaster<ThorTreeDecomposition<Node> > broadcaster;
  ThorTreeDecomposition<Node> decomposition;

  CreateKdBTreeMaster<Point, Node>(param,
      points_channel, nodes_channel, block_size_kb, megs, module,
      n_points, input_cache, points_cache, nodes_cache, &decomposition);

  /* Probably engineered for parallel; shouldn't hurt */
  broadcaster.SetData(decomposition);
  points_cache->Sync();
  nodes_cache->Sync();
  broadcaster.Doit(extra_channel);

  tree_out->Init(param, broadcaster.get(), points_cache, nodes_cache);
}
