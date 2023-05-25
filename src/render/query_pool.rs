use std::{collections::HashMap, ops::Deref, sync::Arc, time::Duration};

use ash::{prelude::VkResult, vk, Device};

pub struct QueryPool {
    query_pool: vk::QueryPool,
    query_count: u32,

    current_idx: u32,
    entries: HashMap<String, u32>,

    device: Arc<Device>,
}

impl QueryPool {
    pub unsafe fn new(
        device: &Arc<Device>,
        query_count: u32,
        query_type: vk::QueryType,
    ) -> VkResult<Self> {
        let query_pool = device.create_query_pool(
            &vk::QueryPoolCreateInfo::default()
                .query_count(query_count)
                .query_type(query_type),
            None,
        )?;

        Ok(Self {
            query_pool,
            query_count,

            current_idx: 0,
            entries: HashMap::new(),

            device: device.clone(),
        })
    }

    #[inline]
    pub unsafe fn write_timestamp(
        &mut self,
        command_buffer: vk::CommandBuffer,
        stage: vk::PipelineStageFlags2,
        name: impl Into<String>,
    ) {
        self.device
            .cmd_write_timestamp2(command_buffer, stage, self.query_pool, self.current_idx);

        self.entries.insert(name.into(), self.current_idx);
        self.current_idx += 1;
    }

    #[inline]
    pub unsafe fn reset(&mut self, command_buffer: vk::CommandBuffer) {
        self.current_idx = 0;
        self.entries.clear();
        self.device
            .cmd_reset_query_pool(command_buffer, self.query_pool, 0, self.query_count);
    }

    #[inline]
    pub unsafe fn get_results(&self) -> VkResult<HashMap<String, Duration>> {
        let mut results = vec![0_u64; self.query_count as usize];

        self.device.get_query_pool_results(
            self.query_pool,
            0,
            &mut results,
            vk::QueryResultFlags::TYPE_64,
        )?;

        Ok(self
            .entries
            .iter()
            .map(|(name, idx)| (name.clone(), Duration::from_nanos(results[*idx as usize])))
            .collect())
    }
}

impl Deref for QueryPool {
    type Target = vk::QueryPool;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.query_pool
    }
}

impl Drop for QueryPool {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_query_pool(self.query_pool, None);
        }
    }
}
